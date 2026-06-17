#!/usr/bin/env python3
"""
Эксперимент 05b: Визуализация траектории сигнала + Микродозинг
Финальная версия: надежный откат весов + экономия памяти (CPU, 16 GB RAM)
"""

import json
import torch
import gc
import os
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # КРИТИЧНО: отключает GUI, экономит сотни МБ памяти
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


def log_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEM {stage}] {mem_mb:.0f} MB")
    return mem_mb


class ActivationTracker:
    """Ловец активаций. Хранит только усредненный вектор последнего токена для каждого слоя."""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # 1. Извлекаем тензор: иногда это кортеж, иногда сразу тензор
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            # 2. Безопасно извлекаем последний токен в зависимости от размерности
            try:
                if tensor.dim() == 3:
                    # Стандартный случай: [batch_size, seq_len, hidden_dim]
                    last_token_act = tensor[0, -1, :].detach().cpu().numpy()
                elif tensor.dim() == 2:
                    # Случай, когда batch-измерение было сжато: [seq_len, hidden_dim]
                    last_token_act = tensor[-1, :].detach().cpu().numpy()
                else:
                    # Аварийный вариант (на всякий случай)
                    flat_tensor = tensor.detach().cpu().flatten().numpy()
                    # Qwen 0.5B имеет hidden_size = 896. Берем последние 896 значений.
                    last_token_act = flat_tensor[-896:]
                
                self.activations[layer_idx] = last_token_act
                
            except Exception as e:
                print(f"  ⚠️ Ошибка на слое {layer_idx}, форма тензора: {getattr(tensor, 'shape', 'unknown')}. Ошибка: {e}")
                # Заполняем нулями, чтобы не ломать весь пайплайн из-за одного слоя
                self.activations[layer_idx] = np.zeros(896)
                
        return hook
    
    def attach(self, model):
        for name, module in model.named_modules():
            # Цепляемся только к основным слоям трансформера (их 24 в Qwen 0.5B)
            if 'model.layers.' in name and name.count('.') == 2:
                try:
                    layer_idx = int(name.split('.')[-1])
                    h = module.register_forward_hook(self._make_hook(layer_idx))
                    self.hooks.append(h)
                except ValueError:
                    pass
        print(f"  🎣 Поймано слоёв: {len(self.hooks)}")
    
    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_trajectory(self):
        if not self.activations:
            return None, []
        sorted_layers = sorted(self.activations.keys())
        trajectory = np.stack([self.activations[i] for i in sorted_layers])
        return trajectory, sorted_layers
    
    def clear(self):
        self.activations = {}
        gc.collect()


def capture_trajectory(model, tokenizer, prompt, device, tracker):
    tracker.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,  # Нам нужен только начальный путь сигнала, не генерируем много
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    del inputs
    gc.collect()
    return tracker.get_trajectory()


def plot_comparison(all_trajectories, save_path):
    """Рисует все дозы на одном графике для сравнения"""
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.plasma(np.linspace(0, 1, len(all_trajectories)))
    
    for (dose, coords, layers), color in zip(all_trajectories, colors):
        ax.plot(coords[:, 0], coords[:, 1], '-', color=color, alpha=0.6, linewidth=2, label=f"dose={dose*100:.2f}%")
        ax.scatter(coords[:, 0], coords[:, 1], c=[color]*len(coords), s=40, zorder=5, edgecolors='black', linewidth=0.3)
        
        # Стрелки направления
        for i in range(len(coords) - 1):
            ax.annotate('', xy=(coords[i+1, 0], coords[i+1, 1]), xytext=(coords[i, 0], coords[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4))
    
    ax.set_title("Сравнение траекторий: как шум искривляет поток сигнала", fontsize=13, pad=15)
    ax.set_xlabel("PC1 (Главная компонента)")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)  # КРИТИЧНО: закрываем фигуру, чтобы освободить память
    gc.collect()


def main():
    print("=" * 60)
    print("Эксперимент 05b: Траектория сигнала + Микродозинг")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports" / "trajectories"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cpu"
    test_prompt = "Calculate: 24578 + 13892 ="  # Простой промпт для быстрой проверки
    print(f"\n📝 Тестовый промпт: {test_prompt}")
    
    print("\n📦 Загрузка модели...")
    log_memory("до модели")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()  # КРИТИЧНО: eval mode отключает dropout и экономит память
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_memory("после модели")
    
    # === ТВОЯ НАДЕЖНАЯ ЛОГИКА СОХРАНЕНИЯ ВЕСОВ ===
    print("💾 Сохраняю оригинальные веса для отката...")
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.clone()
    log_memory("после сохранения весов")
    
    dose_schedule = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03] # Добавил 0.015, чтобы поймать переход
    
    tracker = ActivationTracker()
    tracker.attach(model)
    
    all_trajectories = []
    
    for dose in dose_schedule:
        print(f"\n{'='*60}")
        print(f"🧪 ДОЗА: {dose*100:.2f}%")
        print(f"{'='*60}")
        log_memory(f"перед дозой {dose}")
        
        # 1. Добавляем шум (если доза > 0)
        if dose > 0:
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.randn_like(param) * dose
                    param.add_(noise)
        
        # 2. Захватываем траекторию
        print(f"  🏃 Прогон с трекингом...")
        trajectory, layer_indices = capture_trajectory(model, tokenizer, test_prompt, device, tracker)
        
        if trajectory is None:
            print("  ⚠️ Не удалось захватить траекторию")
            continue
            
        print(f"  📊 Форма траектории: {trajectory.shape}")
        
        # 3. PCA → 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(trajectory)
        variance_ratio = pca.explained_variance_ratio_
        print(f"  🎯 PCA: PC1={variance_ratio[0]*100:.1f}%, PC2={variance_ratio[1]*100:.1f}%")
        
        all_trajectories.append((dose, coords_2d, layer_indices))
        
        # 4. ТВОЯ НАДЕЖНАЯ ЛОГИКА ОТКАТА ВЕСОВ
        if dose > 0:
            print(f"  🔙 Откат шума к оригинальным весам...")
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_weights[name])
        
        tracker.clear()
        gc.collect()
        log_memory(f"после дозы {dose}")
    
    tracker.detach()
    
    print("\n🧹 Освобождение модели...")
    del model
    del tokenizer
    del original_weights
    gc.collect()
    log_memory("после освобождения модели")
    
    # 5. Рисуем финальный график сравнения
    if len(all_trajectories) > 1:
        print("\n🎨 Строю сравнительный график...")
        comparison_path = reports_dir / "comparison_all_doses.png"
        plot_comparison(all_trajectories, comparison_path)
        print(f"💾 Сохранено: {comparison_path}")
        
        # Сохраняем данные для анализа
        metadata = []
        for dose, coords, layers in all_trajectories:
            metadata.append({
                'dose': dose,
                'coords_2d': coords.tolist(),
                'layers': layers
            })
        with open(reports_dir / "trajectories_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✨ ГОТОВО! Смотри файл comparison_all_doses.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
