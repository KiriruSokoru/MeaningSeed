#!/usr/bin/env python3
"""
Глубокий анализ switches: какие нейроны критичны для модели
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Загружаем данные
ckpt = Path("checkpoints")
with open(ckpt / "switches.json") as f:
    data = json.load(f)

switches = data["switches"]

# === АНАЛИЗ 1: Распределение по слоям ===
layer_distribution = {}
for layer_idx, switch_list in switches.items():
    layer_distribution[int(layer_idx)] = len(switch_list)

print("=" * 60)
print("АНАЛИЗ SWITCHES")
print("=" * 60)
print(f"\nВсего слоёв: {len(layer_distribution)}")
print(f"Всего switches: {sum(layer_distribution.values())}")
print(f"Среднее на слой: {np.mean(list(layer_distribution.values())):.1f}")
print(f"Минимум: {min(layer_distribution.values())} (слой {min(layer_distribution, key=layer_distribution.get)})")
print(f"Максимум: {max(layer_distribution.values())} (слой {max(layer_distribution, key=layer_distribution.get)})")

# === АНАЛИЗ 2: Какие нейроны встречаются чаще всего ===
neuron_counter = Counter()
layer_neuron_map = {}

for layer_idx, switch_list in switches.items():
    layer_neuron_map[int(layer_idx)] = set()
    for neuron_idx, score in switch_list:
        neuron_counter[neuron_idx] += 1
        layer_neuron_map[int(layer_idx)].add(neuron_idx)

print(f"\n{'=' * 60}")
print("TOP-20 НЕЙРОНОВ (встречаются в нескольких слоях)")
print("=" * 60)
for neuron, count in neuron_counter.most_common(20):
    print(f"Neuron {neuron:4d}: встречается в {count:2d} слоях")

# === АНАЛИЗ 3: Универсальные нейроны (во всех слоях) ===
universal_neurons = []
for neuron, count in neuron_counter.items():
    if count == len(switches):
        universal_neurons.append(neuron)

print(f"\n{'=' * 60}")
print(f"УНИВЕРСАЛЬНЫЕ НЕЙРОНЫ (во всех {len(switches)} слоях)")
print("=" * 60)
if universal_neurons:
    print(f"Найдено: {len(universal_neurons)} нейронов")
    print(f"Индексы: {sorted(universal_neurons)[:20]}")
else:
    print("Не найдено")

# === АНАЛИЗ 4: Средний score по слоям ===
layer_avg_score = {}
for layer_idx, switch_list in switches.items():
    scores = [score for _, score in switch_list]
    layer_avg_score[int(layer_idx)] = np.mean(scores)

print(f"\n{'=' * 60}")
print("СРЕДНИЙ SCORE ПО СЛОЯМ")
print("=" * 60)
sorted_layers = sorted(layer_avg_score.items())
for layer, score in sorted_layers[:5]:
    print(f"Layer {layer:2d}: avg_score = {score:.4f} (низкие слои)")
print("...")
for layer, score in sorted_layers[-5:]:
    print(f"Layer {layer:2d}: avg_score = {score:.4f} (высокие слои)")

# === ВИЗУАЛИЗАЦИИ ===
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('MeaningSeed Switch Analysis — Qwen2.5-1.5B', fontsize=16, fontweight='bold')

# 1. Распределение по слоям
axes[0, 0].bar(layer_distribution.keys(), layer_distribution.values(), color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Layer Index')
axes[0, 0].set_ylabel('Number of Switches')
axes[0, 0].set_title('Switch Distribution Across Layers')
axes[0, 0].grid(True, alpha=0.3)

# 2. Top нейроны
top_neurons = neuron_counter.most_common(30)
neuron_ids = [n for n, c in top_neurons]
neuron_counts = [c for n, c in top_neurons]
axes[0, 1].barh(neuron_ids, neuron_counts, color='coral', alpha=0.7)
axes[0, 1].set_xlabel('Frequency (layers)')
axes[0, 1].set_ylabel('Neuron Index')
axes[0, 1].set_title('Top-30 Most Frequent Neurons')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Средний score по слоям
axes[1, 0].plot(layer_avg_score.keys(), layer_avg_score.values(), 'o-', color='green', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Layer Index')
axes[1, 0].set_ylabel('Average Score')
axes[1, 0].set_title('Average Switch Score by Layer')
axes[1, 0].grid(True, alpha=0.3)

# 4. Heatmap: какие нейроны в каких слоях
heatmap_data = np.zeros((len(switches), 1536))
for layer_idx, switch_list in switches.items():
    for neuron_idx, score in switch_list:
        heatmap_data[int(layer_idx), neuron_idx] = 1

axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='Blues', interpolation='none')
axes[1, 1].set_xlabel('Neuron Index')
axes[1, 1].set_ylabel('Layer Index')
axes[1, 1].set_title('Switch Neuron Heatmap')

plt.tight_layout()
plt.savefig(ckpt / 'switch_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Визуализация сохранена: {ckpt / 'switch_analysis.png'}")

# === СТАТИСТИКА ===
print(f"\n{'=' * 60}")
print("СТАТИСТИКА")
print("=" * 60)
print(f"Уникальных нейронов: {len(neuron_counter)}")
print(f"Переиспользование: {100 * (1 - len(neuron_counter) / sum(layer_distribution.values())):.1f}%")
print(f"Средняя важность нейрона: {np.mean(list(neuron_counter.values())):.2f} слоёв")

# Сохраняем результаты анализа
analysis_results = {
    "total_layers": len(switches),
    "total_switches": sum(layer_distribution.values()),
    "unique_neurons": len(neuron_counter),
    "universal_neurons": len(universal_neurons),
    "top_neurons": [{"neuron": n, "count": c} for n, c in neuron_counter.most_common(50)],
    "layer_distribution": layer_distribution,
    "layer_avg_score": layer_avg_score,
    "reuse_percentage": 100 * (1 - len(neuron_counter) / sum(layer_distribution.values()))
}

with open(ckpt / "switch_analysis.json", 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f"✓ Детальный анализ сохранён: {ckpt / 'switch_analysis.json'}")
