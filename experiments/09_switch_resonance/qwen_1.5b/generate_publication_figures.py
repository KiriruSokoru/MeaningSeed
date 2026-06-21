"""
Генерация всех визуализаций для публикации на Habr
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')
from pathlib import Path

# Создаём папку для фигур
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

# === ЗАГРУЖАЕМ ДАННЫЕ ===
with open("checkpoints/switches.json") as f:
    switches_data = json.load(f)

with open("checkpoints/switch_analysis.json") as f:
    analysis_data = json.load(f)

# === ФИГУРА 1: PPL vs Noise при разных top_k ===
fig, ax = plt.subplots(figsize=(10, 6))

noise_levels = [0.01, 0.02, 0.03, 0.05]
top_k_values = [400, 1000, 2000]

# Данные из экспериментов
results = {
    400: {0.01: 460.86, 0.02: 16218817.0, 0.05: 484123982430208.0},
    1000: {0.02: 4254467.5},
    2000: {0.01: 50.98, 0.02: 232.04, 0.03: 2715.95, 0.05: 19490582528.0}
}

colors = ['#2196F3', '#FF9800', '#4CAF50']
for i, top_k in enumerate(top_k_values):
    if top_k in results:
        x = list(results[top_k].keys())
        y = list(results[top_k].values())
        ax.semilogy(x, y, 'o-', color=colors[i], linewidth=2.5, 
                   markersize=10, label=f'top_k={top_k}')

ax.axhline(y=12.41, color='red', linestyle='--', linewidth=2, label='Baseline PPL (12.41)')
ax.set_xlabel('Noise Standard Deviation', fontsize=13, fontweight='bold')
ax.set_ylabel('Perplexity (log scale)', fontsize=13, fontweight='bold')
ax.set_title('Weight Restoration: PPL vs Noise Level\nQwen2.5-1.5B', 
             fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(1, 1e16)

plt.tight_layout()
plt.savefig(fig_dir / 'fig1_ppl_vs_noise.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1: PPL vs Noise")

# === ФИГУРА 2: Score по слоям ===
fig, ax = plt.subplots(figsize=(10, 6))

layers = [int(k) for k in analysis_data['layer_avg_score'].keys()]
scores = [float(v) for v in analysis_data['layer_avg_score'].values()]

# Линейная регрессия
z = np.polyfit(layers, scores, 1)
p = np.poly1d(z)

ax.plot(layers, scores, 'o-', color='#E91E63', linewidth=2.5, markersize=8, 
        label='Average Switch Score')
ax.plot(layers, p(layers), '--', color='gray', linewidth=2, 
        label=f'Linear fit: y = {z[0]:.3f}x + {z[1]:.3f}')

ax.fill_between(layers, scores, alpha=0.2, color='#E91E63')
ax.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Switch Score', fontsize=13, fontweight='bold')
ax.set_title('Switch Importance Grows with Layer Depth\nQwen2.5-1.5B', 
             fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Аннотации
ax.annotate('Input layers\n(low importance)', xy=(0, scores[0]), 
            xytext=(3, 1), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('Output layers\n(high importance)', xy=(27, scores[-1]), 
            xytext=(22, 4.5), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig(fig_dir / 'fig2_score_by_layer.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2: Score by Layer")

# === ФИГУРА 3: Heatmap switches ===
fig, ax = plt.subplots(figsize=(14, 8))

switches = switches_data['switches']
heatmap = np.zeros((28, 1536))

for layer_idx, switch_list in switches.items():
    for neuron_idx, score in switch_list:
        heatmap[int(layer_idx), neuron_idx] = score

im = ax.imshow(heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_xlabel('Neuron Index (0-1535)', fontsize=13, fontweight='bold')
ax.set_ylabel('Layer Index (0-27)', fontsize=13, fontweight='bold')
ax.set_title('Switch Neuron Heatmap: Score Distribution\n'
             'Vertical stripes = meta-switches (important across all layers)',
             fontsize=14, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Switch Score', fontsize=12, fontweight='bold')

# Выделяем мета-переключатели
meta_neurons = [1215, 102, 1173, 872, 1412, 335, 1214, 1222, 1248, 251]
for n in meta_neurons[:5]:
    ax.axvline(x=n, color='red', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(fig_dir / 'fig3_switch_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3: Switch Heatmap")

# === ФИГУРА 4: Top нейронов ===
fig, ax = plt.subplots(figsize=(10, 8))

top_neurons = analysis_data['top_neurons'][:20]
neurons = [n['neuron'] for n in top_neurons]
counts = [n['count'] for n in top_neurons]

y_pos = np.arange(len(neurons))
bars = ax.barh(y_pos, counts, color='#3F51B5', alpha=0.8, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Neuron {n}' for n in neurons], fontsize=10)
ax.set_xlabel('Appears in (layers)', fontsize=13, fontweight='bold')
ax.set_title('Top-20 Meta-Switches\nNeurons critical across ALL 28 layers',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 30)
ax.grid(True, alpha=0.3, axis='x')

# Добавляем значения
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax.text(count + 0.3, bar.get_y() + bar.get_height()/2, 
            f'{count}/28', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'fig4_top_neurons.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4: Top Neurons")

# === ФИГУРА 5: Efficiency plot ===
fig, ax = plt.subplots(figsize=(10, 6))

params_restored = [78400, 196000, 301056]
improvement = [7.0, 62.1, 203509.2]
labels = ['top_k=400', 'top_k=1000', 'top_k=2000']

ax.semilogy(params_restored, improvement, 'o-', color='#009688', 
           linewidth=3, markersize=12)

for i, (p, imp, label) in enumerate(zip(params_restored, improvement, labels)):
    ax.annotate(f'{label}\n{imp:.1f}x', 
                xy=(p, imp), xytext=(10, 10),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Restored Parameters', fontsize=13, fontweight='bold')
ax.set_ylabel('Improvement Factor (log scale)', fontsize=13, fontweight='bold')
ax.set_title('Restoration Efficiency: Params vs Improvement\n'
             'Qwen2.5-1.5B, noise_std=0.02',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 350000)

plt.tight_layout()
plt.savefig(fig_dir / 'fig5_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Figure 5: Efficiency")

# === ФИГУРА 6: Summary infographic ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MeaningSeed: Topological Seeds in LLMs\nKey Findings', 
             fontsize=16, fontweight='bold', y=0.98)

# 6.1: Params restored
ax = axes[0, 0]
categories = ['Total Model', 'Restored']
values = [1.5e9, 301056]
colors = ['#BDBDBD', '#4CAF50']
bars = ax.bar(categories, values, color=colors, edgecolor='black')
ax.set_ylabel('Parameters')
ax.set_title('0.02% of Parameters\nRestore 200,000x Improvement', fontweight='bold')
ax.set_yscale('log')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
            f'{val:.0e}', ha='center', fontweight='bold')

# 6.2: Noise threshold
ax = axes[0, 1]
noise = [0.01, 0.02, 0.03, 0.05]
ppl_after = [50.98, 232.04, 2715.95, 19490582528.0]
ax.semilogy(noise, ppl_after, 'o-', color='#F44336', linewidth=2.5, markersize=10)
ax.axvline(x=0.04, color='orange', linestyle='--', linewidth=2, label='Critical threshold')
ax.set_xlabel('Noise Level')
ax.set_ylabel('Restored PPL')
ax.set_title('Critical Noise Threshold ≈ 0.04', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 6.3: Layer importance
ax = axes[1, 0]
layer_groups = ['Layers 0-9\n(Input)', 'Layers 10-19\n(Middle)', 'Layers 20-27\n(Output)']
avg_scores = [0.95, 1.09, 3.95]
colors = ['#2196F3', '#FFC107', '#F44336']
bars = ax.bar(layer_groups, avg_scores, color=colors, edgecolor='black')
ax.set_ylabel('Average Switch Score')
ax.set_title('Upper Layers = 4x More Important', fontweight='bold')
for bar, val in zip(bars, avg_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}', ha='center', fontweight='bold')

# 6.4: Meta-switches
ax = axes[1, 1]
ax.bar(['Meta-switches', 'Other neurons'], [50, 1486], 
       color=['#9C27B0', '#9E9E9E'], edgecolor='black')
ax.set_ylabel('Number of Neurons')
ax.set_title('50 Meta-Switches\nCritical in ALL 28 layers', fontweight='bold')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(fig_dir / 'fig6_summary.png', dpi=300, bbox_inches='tight')
print("✓ Figure 6: Summary")

print(f"\n✅ Все 6 фигур сохранены в {fig_dir}/")
print("Готовы для публикации!")
