#!/usr/bin/env python3
"""
Эксперимент 06: Прицельная хирургия (Targeted Surgery)
Внедрение "Мастеров" в критические слои (19-23) для восстановления траектории

Стратегия:
1. Найти топ-нейроны в слоях 19-23 (мастера)
2. Добавить шум к модели
3. Усилить мастеров (вернуть их к оригинальным весам)
4. Проверить, вернулась ли траектория к эталону
"""

import json
import torch
import gc
import os
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    """Ловец активаций"""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            try:
                if tensor.dim() == 3:
                    last_token_act = tensor[0, -1, :].detach().cpu().numpy()
                elif tensor.dim() == 2:
                    last_token_act = tensor[-1, :].detach().cpu().numpy()
                else:
                    flat_tensor = tensor.detach().cpu().flatten().numpy()
                    last_token_act = flat_tensor[-896:]
                
                self.activations[layer_idx] = last_token_act
            except Exception as e:
                print(f"  ⚠️ Ошибка на слое {layer_idx}: {e}")
                self.activations[layer_idx] = np.zeros(896)
                
        return hook
    
    def attach(self, model):
        for name, module in model.named_modules():
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
    """Один прогон с трекингом"""
    tracker.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    del inputs
    gc.collect()
    return tracker.get_trajectory()


def find_masters_in_layers(model, tokenizer, prompt, device, target_layers, top_k=10):
    """
    Находит топ-K нейронов (мастеров) в целевых слоях
    по их активности на тестовом промпте
    """
    print(f"\n🔍 Поиск мастеров в слоях {target_layers}...")
    
    masters = {}  # {layer_idx: [neuron_indices]}
    
    # Хук для сбора активаций всех нейронов
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            if tensor.dim() == 3:
                # [batch, seq_len, hidden_dim] -> берем последний токен
                act = tensor[0, -1, :].detach().cpu().numpy()
            elif tensor.dim() == 2:
                act = tensor[-1, :].detach().cpu().numpy()
            else:
                act = tensor.detach().cpu().flatten().numpy()[-896:]
            
            # Находим топ-K нейронов по абсолютной активности
            top_neurons = np.argsort(np.abs(act))[-top_k:][::-1]
            masters[layer_idx] = top_neurons.tolist()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if 'model.layers.' in name and name.count('.') == 2:
            try:
                layer_idx = int(name.split('.')[-1])
                if layer_idx in target_layers:
                    h = module.register_forward_hook(make_hook(layer_idx))
                    hooks.append(h)
            except ValueError:
                pass
    
    # Прогон для поиска мастеров
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    del inputs
    for h in hooks:
        h.remove()
    gc.collect()
    
    print(f"  ✅ Найдено мастеров: {sum(len(v) for v in masters.values())}")
    for layer, neurons in sorted(masters.items()):
        print(f"    Layer {layer}: {neurons[:5]}...")
    
    return masters


def apply_surgery(model, original_weights, masters, amplification=1.5):
    """
    Применяет хирургию: усиливает мастеров в целевых слоях
    Возвращает их к оригинальным весам (или усиливает)
    """
    print(f"\n🔧 Применение хирургии (усиление ×{amplification})...")
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Ищем параметры слоев из masters
            for layer_idx, neuron_indices in masters.items():
                # Формат имени: "model.layers.X.mlp.down_proj.weight"
                if f"model.layers.{layer_idx}." in name:
                    # Усиливаем только выбранные нейроны
                    if param.dim() == 2:  # Матрица весов
                        # Для down_proj: [hidden, hidden]
                        # Усиливаем строки, соответствующие мастерам
                        for neuron_idx in neuron_indices:
                            if neuron_idx < param.shape[0]:
                                # Возвращаем к оригинальному весу и усиливаем
                                original_val = original_weights[name][neuron_idx, :].clone()
                                param[neuron_idx, :] = original_val * amplification
                    elif param.dim() == 1:  # Bias
                        for neuron_idx in neuron_indices:
                            if neuron_idx < param.shape[0]:
                                original_val = original_weights[name][neuron_idx].clone()
                                param[neuron_idx] = original_val * amplification
    
    print(f"  ✅ Хирургия применена к {sum(len(v) for v in masters.values())} нейронам")


def plot_comparison_3way(trajectories, save_path):
    """Рисует сравнение трех траекторий: Эталон, Шум, Восстановленная"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {
        'Эталон (0.0%)': 'black',
        'Шум (1.5%)': 'red',
        'Восстановленная': 'green'
    }
    
    for label, (dose, coords, layers) in trajectories.items():
        color = colors.get(label, 'blue')
        linestyle = '--' if 'Эталон' in label else '-'
        linewidth = 2.5 if 'Восстановленная' in label else 1.5
        
        ax.plot(coords[:, 0], coords[:, 1], 
                color=color, linestyle=linestyle, linewidth=linewidth, 
                label=label, zorder=5)
        
        # Точки
        ax.scatter(coords[:, 0], coords[:, 1], 
                   c=[color]*len(coords), s=40, zorder=6, 
                   edgecolors='black', linewidth=0.3)
        
        # Стрелки
        for i in range(len(coords) - 1):
            ax.annotate('', 
                        xy=(coords[i+1, 0], coords[i+1, 1]),
                        xytext=(coords[i, 0], coords[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4))
    
    ax.set_title("Прицельная хирургия: Возврат к эталону", fontsize=14, pad=15)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


def main():
    print("=" * 60)
    print("Эксперимент 06: Прицельная хирургия")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cpu"
    test_prompt = "Calculate: 24578 + 13892 ="
    print(f"\n📝 Тестовый промпт: {test_prompt}")
    
    # === 1. Загрузка модели ===
    print("\n📦 Загрузка модели...")
    log_memory("до модели")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_memory("после модели")
    
    # === 2. Сохранение оригинальных весов ===
    print("💾 Сохранение оригинальных весов...")
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.clone()
    log_memory("после сохранения весов")
    
    # === 3. Захват эталонной траектории ===
    print("\n🎯 Захват эталонной траектории (0% шума)...")
    tracker = ActivationTracker()
    tracker.attach(model)
    
    ref_trajectory, ref_layers = capture_trajectory(model, tokenizer, test_prompt, device, tracker)
    print(f"  ✅ Эталон захвачен: {ref_trajectory.shape}")
    
    # === 4. Добавление шума (1.5% — критическая точка из эксперимента 05) ===
    noise_dose = 0.015
    print(f"\n️ Добавление шума {noise_dose*100}%...")
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_dose
            param.add_(noise)
    
    noisy_trajectory, _ = capture_trajectory(model, tokenizer, test_prompt, device, tracker)
    print(f"  ✅ Шумная траектория захвачена: {noisy_trajectory.shape}")
    
    # === 5. Поиск мастеров в критических слоях (19-23) ===
    # Сначала восстановим веса для честного поиска мастеров
    print("\n🔙 Восстановление весов для поиска мастеров...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(original_weights[name])
    
    target_layers = [19, 20, 21, 22, 23]
    masters = find_masters_in_layers(model, tokenizer, test_prompt, device, target_layers, top_k=10)
    
    # === 6. Применение хирургии ===
    # Сначала добавляем шум обратно
    print(f"\n️ Повторное добавление шума {noise_dose*100}%...")
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_dose
            param.add_(noise)
    
    # Теперь применяем хирургию
    apply_surgery(model, original_weights, masters, amplification=1.5)
    
    # === 7. Захват восстановленной траектории ===
    print("\n Захват восстановленной траектории...")
    restored_trajectory, _ = capture_trajectory(model, tokenizer, test_prompt, device, tracker)
    print(f"  ✅ Восстановленная траектория захвачена: {restored_trajectory.shape}")
    
    # === 8. PCA и визуализация ===
    print("\n🎨 Построение графиков...")
    tracker.detach()
    
    # PCA для всех трех траекторий
    all_trajectories = np.vstack([ref_trajectory, noisy_trajectory, restored_trajectory])
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(all_trajectories)
    
    # Разделяем обратно
    n_layers = len(ref_layers)
    ref_coords = all_coords[:n_layers]
    noisy_coords = all_coords[n_layers:2*n_layers]
    restored_coords = all_coords[2*n_layers:]
    
    trajectories = {
        'Эталон (0.0%)': (0.0, ref_coords, ref_layers),
        'Шум (1.5%)': (noise_dose, noisy_coords, ref_layers),
        'Восстановленная': (noise_dose, restored_coords, ref_layers)
    }
    
    # Сохраняем данные
    metadata = []
    for label, (dose, coords, layers) in trajectories.items():
        metadata.append({
            'label': label,
            'dose': dose,
            'coords_2d': coords.tolist(),
            'layers': layers
        })
    
    with open(reports_dir / "surgery_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Рисуем график
    plot_comparison_3way(trajectories, reports_dir / "surgery_comparison.png")
    print(f"  ✅ График сохранен: {reports_dir / 'surgery_comparison.png'}")
    
    # === 9. Анализ результатов ===
    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # Считаем отклонения от эталона
    noisy_deviation = np.mean(np.linalg.norm(noisy_coords - ref_coords, axis=1))
    restored_deviation = np.mean(np.linalg.norm(restored_coords - ref_coords, axis=1))
    
    print(f"\nСреднее отклонение от эталона:")
    print(f"  Шумная модель:      {noisy_deviation:.2f} единиц")
    print(f"  Восстановленная:    {restored_deviation:.2f} единиц")
    
    if restored_deviation < noisy_deviation:
        improvement = ((noisy_deviation - restored_deviation) / noisy_deviation) * 100
        print(f"\n✅ УЛУЧШЕНИЕ: {improvement:.1f}%")
        print("   Хирургия сработала! Траектория вернулась к эталону.")
    else:
        print(f"\n❌ УХУДШЕНИЕ: Хирургия не помогла.")
    
    # Освобождение памяти
    print("\n🧹 Освобождение памяти...")
    del model
    del tokenizer
    del original_weights
    gc.collect()
    log_memory("после освобождения")
    
    print("\n" + "=" * 60)
    print("✨ ЭКСПЕРИМЕНТ 06 ЗАВЕРШЕН!")
    print("=" * 60)
    print(f"📂 Результаты в: {reports_dir}")
    print("\n🔍 ЧТО СМОТРЕТЬ:")
    print("  1. surgery_comparison.png — сравнение трех траекторий")
    print("  2. surgery_metadata.json — сырые данные")
    print("\n💡 ИНТЕРПРЕТАЦИЯ:")
    print("  • Если зеленая линия (восстановленная) ближе к черной (эталон),")
    print("    чем красная (шумная) — хирургия работает!")
    print("  • Это доказывает, что 'Мастера' могут компенсировать шум.")


if __name__ == "__main__":
    main()
