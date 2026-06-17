#!/usr/bin/env python3
"""
Эксперимент 07 v4: Поток Риччи (Ricci Flow Cap)
ПРАВИЛЬНАЯ ВЕРСИЯ: заменяем сингулярности на ЭТАЛОННЫЕ значения (колпачок Перельмана)

Логика:
1. Захватываем эталонные активации (0% шума)
2. Добавляем шум 1.5%
3. Захватываем зашумленные активации
4. Находим сингулярности в слоях 19-23 (нейроны с максимальным отклонением)
5. Применяем колпачок: заменяем активации сингулярностей на эталонные значения
6. Захватываем все 24 слоя после коррекции
7. Сравниваем три траектории через PCA
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


def extract_trajectory(activations):
    """Извлекает траекторию из активаций (последний токен каждого слоя)"""
    trajectory = []
    for layer_idx in sorted(activations.keys()):
        act = activations[layer_idx]
        if act.ndim == 3:
            last_token = act[0, -1, :].numpy()
        elif act.ndim == 2:
            last_token = act[-1, :].numpy()
        else:
            last_token = act.flatten().numpy()[-896:]
        trajectory.append(last_token)
    return np.stack(trajectory)


def find_singularities(reference_acts, noisy_acts, top_k=10):
    """Находит нейроны с максимальным отклонением от эталона"""
    ref_last = reference_acts[0, -1, :].numpy()
    noisy_last = noisy_acts[0, -1, :].numpy()
    
    deviations = np.abs(noisy_last - ref_last)
    singularities = np.argsort(deviations)[-top_k:][::-1].tolist()
    
    return singularities


def apply_ricci_cap_and_capture(model, tokenizer, prompt, device, 
                                 reference_acts, noisy_acts, 
                                 target_layers, top_k=10):
    """
    Применяет колпачок Перельмана и захватывает ВСЕ активации после коррекции.
    
    Ключевая идея: заменяем активации сингулярных нейронов на их ЭТАЛОННЫЕ значения.
    Это и есть "вшивание гладкого колпачка" вместо сингулярности.
    """
    print(f"\n🔧 Применение колпачка Перельмана (эталонная замена)...")
    
    # Находим сингулярности для каждого целевого слоя
    layer_singularities = {}
    for layer_idx in target_layers:
        if layer_idx in reference_acts and layer_idx in noisy_acts:
            singularities = find_singularities(
                reference_acts[layer_idx],
                noisy_acts[layer_idx],
                top_k
            )
            layer_singularities[layer_idx] = singularities
            print(f"  Layer {layer_idx}: {len(singularities)} сингулярностей")
    
    # Словарь для захвата всех активаций
    corrected_activations = {}
    
    # Создаем хуки
    all_hooks = []
    
    for name, module in model.named_modules():
        if 'model.layers.' in name and name.count('.') == 2:
            try:
                layer_idx = int(name.split('.')[-1])
                
                # 1. Корректирующий хук (если слой целевой)
                if layer_idx in layer_singularities:
                    singularities = layer_singularities[layer_idx]
                    # Получаем эталонные активации для этого слоя (через default arg)
                    ref_tensor = reference_acts[layer_idx]
                    
                    def make_correction_hook(sing_indices, ref_t):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                tensor = output[0]
                            else:
                                tensor = output
                            
                            # Модифицируем in-place: заменяем сингулярности на эталон
                            with torch.no_grad():
                                if tensor.ndim == 3:
                                    for sing_idx in sing_indices:
                                        tensor[0, -1, sing_idx] = ref_t[0, -1, sing_idx]
                            return tensor
                        return hook
                    
                    h_corr = module.register_forward_hook(
                        make_correction_hook(singularities, ref_tensor)
                    )
                    all_hooks.append(h_corr)
                
                # 2. Хук захвата (всегда, для всех слоёв)
                # Регистрируем ПОСЛЕ корректирующего, чтобы захватить модифицированные активации
                def make_capture_hook(idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            tensor = output[0]
                        else:
                            tensor = output
                        corrected_activations[idx] = tensor.detach().cpu().clone()
                    return hook
                
                h_cap = module.register_forward_hook(make_capture_hook(layer_idx))
                all_hooks.append(h_cap)
                
            except ValueError:
                pass
    
    print(f"  🎣 Зарегистрировано хуков: {len(all_hooks)}")
    
    # Запускаем модель с хуками
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    del inputs
    for h in all_hooks:
        h.remove()
    gc.collect()
    
    return corrected_activations


def plot_comparison(trajectories_dict, save_path):
    """Рисует сравнение траекторий"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {
        'Эталон': 'black',
        'Шум 1.5%': 'red',
        'Поток Риччи': 'green'
    }
    
    for label, coords in trajectories_dict.items():
        color = colors.get(label, 'blue')
        linestyle = '--' if 'Эталон' in label else '-'
        linewidth = 2.5 if 'Риччи' in label else 1.5
        
        ax.plot(coords[:, 0], coords[:, 1],
                color=color, linestyle=linestyle, linewidth=linewidth,
                label=label, zorder=5)
        
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=[color]*len(coords), s=40, zorder=6,
                   edgecolors='black', linewidth=0.3)
        
        for i in range(len(coords) - 1):
            ax.annotate('',
                        xy=(coords[i+1, 0], coords[i+1, 1]),
                        xytext=(coords[i, 0], coords[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4))
    
    ax.set_title("Эксперимент 07 v4: Поток Риччи (эталонная замена)", fontsize=14, pad=15)
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
    print("Эксперимент 07 v4: Поток Риччи (эталонная замена)")
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
    
    # === 2. Захват эталонных активаций ===
    print("\n🎯 Захват эталонных активаций (0% шума)...")
    
    reference_acts = {}
    def make_ref_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            reference_acts[layer_idx] = tensor.detach().cpu().clone()
        return hook
    
    ref_hooks = []
    for name, module in model.named_modules():
        if 'model.layers.' in name and name.count('.') == 2:
            try:
                layer_idx = int(name.split('.')[-1])
                h = module.register_forward_hook(make_ref_hook(layer_idx))
                ref_hooks.append(h)
            except ValueError:
                pass
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    del inputs
    for h in ref_hooks:
        h.remove()
    gc.collect()
    
    ref_trajectory = extract_trajectory(reference_acts)
    print(f"  ✅ Эталон захвачен: {ref_trajectory.shape}")
    
    # === 3. Добавление шума ===
    noise_dose = 0.015
    print(f"\n⚡ Добавление шума {noise_dose*100}%...")
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_dose
            param.add_(noise)
    
    # === 4. Захват зашумленных активаций ===
    print("\n📊 Захват зашумленных активаций...")
    
    noisy_acts = {}
    def make_noisy_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            noisy_acts[layer_idx] = tensor.detach().cpu().clone()
        return hook
    
    noisy_hooks = []
    for name, module in model.named_modules():
        if 'model.layers.' in name and name.count('.') == 2:
            try:
                layer_idx = int(name.split('.')[-1])
                h = module.register_forward_hook(make_noisy_hook(layer_idx))
                noisy_hooks.append(h)
            except ValueError:
                pass
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    del inputs
    for h in noisy_hooks:
        h.remove()
    gc.collect()
    
    noisy_trajectory = extract_trajectory(noisy_acts)
    print(f"  ✅ Шум захвачен: {noisy_trajectory.shape}")
    
    # === 5. Применение колпачка Перельмана ===
    target_layers = [19, 20, 21, 22, 23]
    print(f"\n🔧 Применение колпачка Перельмана к слоям {target_layers}...")
    
    corrected_acts = apply_ricci_cap_and_capture(
        model, tokenizer, test_prompt, device,
        reference_acts, noisy_acts,
        target_layers, top_k=10
    )
    
    corrected_trajectory = extract_trajectory(corrected_acts)
    print(f"  ✅ Колпачок применен, траектория захвачена: {corrected_trajectory.shape}")
    
    # === 6. PCA и визуализация ===
    print("\n🎨 Построение графиков...")
    
    # Объединяем все траектории для PCA
    all_trajectories = np.vstack([ref_trajectory, noisy_trajectory, corrected_trajectory])
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(all_trajectories)
    
    n_layers = len(reference_acts)
    ref_coords = all_coords[:n_layers]
    noisy_coords = all_coords[n_layers:2*n_layers]
    corrected_coords = all_coords[2*n_layers:]
    
    trajectories_dict = {
        'Эталон': ref_coords,
        'Шум 1.5%': noisy_coords,
        'Поток Риччи': corrected_coords
    }
    
    # Сохраняем данные
    metadata = []
    for label, coords in trajectories_dict.items():
        metadata.append({
            'label': label,
            'coords_2d': coords.tolist(),
            'layers': list(range(n_layers))
        })
    
    with open(reports_dir / "ricci_metadata_v4.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Рисуем график
    plot_comparison(trajectories_dict, reports_dir / "ricci_flow_comparison_v4.png")
    print(f"  ✅ График сохранен: {reports_dir / 'ricci_flow_comparison_v4.png'}")
    
    # === 7. Анализ результатов ===
    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    noisy_deviation = np.mean(np.linalg.norm(noisy_coords - ref_coords, axis=1))
    corrected_deviation = np.mean(np.linalg.norm(corrected_coords - ref_coords, axis=1))
    
    print(f"\nСреднее отклонение от эталона:")
    print(f"  Шумная модель:    {noisy_deviation:.2f} единиц")
    print(f"  Поток Риччи:      {corrected_deviation:.2f} единиц")
    
    if corrected_deviation < noisy_deviation:
        improvement = ((noisy_deviation - corrected_deviation) / noisy_deviation) * 100
        print(f"\n✅ УЛУЧШЕНИЕ: {improvement:.1f}%")
        print("   Колпачок Перельмана сработал! Траектория вернулась к эталону.")
    else:
        print(f"\n❌ УХУДШЕНИЕ: Колпачок не помог.")
        worsening = ((corrected_deviation - noisy_deviation) / noisy_deviation) * 100
        print(f"   Отклонение увеличилось на {worsening:.1f}%")
    
    # Также считаем отклонение по слоям 19-23 отдельно
    print(f"\n🔍 Отклонение в целевых слоях (19-23):")
    noisy_dev_19_23 = np.mean(np.linalg.norm(noisy_coords[19:24] - ref_coords[19:24], axis=1))
    corrected_dev_19_23 = np.mean(np.linalg.norm(corrected_coords[19:24] - ref_coords[19:24], axis=1))
    print(f"  Шум (слои 19-23):    {noisy_dev_19_23:.2f} единиц")
    print(f"  Риччи (слои 19-23):  {corrected_dev_19_23:.2f} единиц")
    
    if corrected_dev_19_23 < noisy_dev_19_23:
        improvement_local = ((noisy_dev_19_23 - corrected_dev_19_23) / noisy_dev_19_23) * 100
        print(f"  ✅ Локальное улучшение: {improvement_local:.1f}%")
    
    # Освобождение памяти
    print("\n🧹 Освобождение памяти...")
    del model
    del tokenizer
    del reference_acts
    del noisy_acts
    del corrected_acts
    gc.collect()
    log_memory("после освобождения")
    
    print("\n" + "=" * 60)
    print("✨ ЭКСПЕРИМЕНТ 07 v4 ЗАВЕРШЕН!")
    print("=" * 60)
    print(f"📂 Результаты в: {reports_dir}")
    print("\n ЧТО СМОТРЕТЬ:")
    print("  1. ricci_flow_comparison_v4.png — сравнение трех траекторий")
    print("  2. ricci_metadata_v4.json — сырые данные")
    print("\n💡 ИНТЕРПРЕТАЦИЯ:")
    print("  • Если зеленая линия (Поток Риччи) ближе к черной (Эталон),")
    print("    чем красная (Шум) — топологическая хирургия работает!")
    print("  • Особенно важно отклонение в слоях 19-23 — там мы применяли колпачок.")


if __name__ == "__main__":
    main()
