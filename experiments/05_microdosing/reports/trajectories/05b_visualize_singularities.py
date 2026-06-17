#!/usr/bin/env python3
"""
Эксперимент 05c: Карта Сингулярностей
Визуализация отклонения от эталона (0.0%)
Показывает, где именно нужно применять "Мастеров" (MeaningSeed).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def load_data():
    if not os.path.exists("trajectories_metadata.json"):
        print("Ошибка: файл trajectories_metadata.json не найден!")
        return None
    
    with open("trajectories_metadata.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_value(item, key_variants):
    """Получает значение по одному из возможных ключей (с пробелами или без)"""
    for key in key_variants:
        if key in item:
            return item[key]
    return None

def main():
    data = load_data()
    if data is None:
        return

    print("🔍 Анализируем данные...")
    
    # Находим Эталон (Dose 0.0)
    ref_dose_data = None
    for item in data:
        dose_val = get_value(item, ["dose", "dose ", "dose  "])
        if dose_val is not None and float(dose_val) == 0.0:
            ref_dose_data = item
            break
    
    if not ref_dose_data:
        print("❌ Не найден эталонный проход (dose 0.0)!")
        print("Доступные дозы:")
        for item in data:
            print(f"  - {get_value(item, ['dose', 'dose ', 'dose  '])}")
        return

    ref_coords = np.array(get_value(ref_dose_data, ["coords_2d", "coords_2d ", "coords_2d  "]))
    ref_layers = get_value(ref_dose_data, ["layers", "layers ", "layers  "])
    
    print(f"✅ Эталон найден. Слоёв: {len(ref_layers)}, Форма: {ref_coords.shape}")

    # Готовим графики
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Рисуем Эталон (Пунктир)
    ax.plot(ref_coords[:, 0], ref_coords[:, 1], 
            color='black', linestyle='--', linewidth=2, label='Эталон (0.0%)', zorder=10)
    
    # Рисуем точки Эталона
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], 
               c='black', s=30, marker='x', zorder=11)

    # Проходим по всем остальным дозам
    sorted_data = sorted(data, key=lambda x: float(get_value(x, ["dose", "dose ", "dose  "]) or 0))

    max_deviations_per_layer = np.zeros(len(ref_layers))

    for item in sorted_data:
        dose_val = float(get_value(item, ["dose", "dose ", "dose  "]) or 0)
        if dose_val == 0.0:
            continue
            
        noisy_coords = np.array(get_value(item, ["coords_2d", "coords_2d ", "coords_2d  "]))
        
        # Выбираем цвет из палитры
        color = plt.cm.viridis(min(dose_val / 0.04, 1.0))
        
        # Рисуем траекторию шума
        ax.plot(noisy_coords[:, 0], noisy_coords[:, 1], 
                color=color, linestyle='-', alpha=0.7, linewidth=1.5, 
                label=f'Шум {dose_val*100:.1f}%', zorder=2)
        
        # === ГЛАВНАЯ МАГИЯ: Рисуем "Ребра Сингулярности" ===
        layer_indices = get_value(item, ["layers", "layers ", "layers  "]) or ref_layers
        
        for i, layer in enumerate(layer_indices):
            if i >= len(ref_coords):
                break
            
            p_ref = ref_coords[i]
            p_noise = noisy_coords[i]
            
            # Считаем расстояние
            dist = np.linalg.norm(p_ref - p_noise)
            
            # Суммируем отклонения
            max_deviations_per_layer[i] += dist
            
            # Рисуем линию-соединитель (если отклонение значимое)
            if dist > 5.0:
                ax.plot([p_ref[0], p_noise[0]], [p_ref[1], p_noise[1]], 
                        color='red', alpha=0.3, linewidth=1, zorder=1)
                
                # На концах линий (у шума) ставим точку
                ax.scatter([p_noise[0]], [p_noise[1]], 
                           c=[color], s=10, zorder=2)

    # Настройки графика
    ax.set_title("Карта Сингулярностей: Отклонение от Эталона", fontsize=16, pad=20)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Сохраняем
    plt.tight_layout()
    output_path = "singularity_map.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ График сохранен: {output_path}")

    # === АНАЛИЗ: ГДЕ САМЫЕ СИЛЬНЫЕ СИНГУЛЯРНОСТИ? ===
    print("\n" + "="*60)
    print("🔥 ГДЕ НУЖНЫ МАСТЕРА? (Топ-5 слоёв с максимальным отклонением)")
    print("="*60)
    
    top_layers_indices = np.argsort(max_deviations_per_layer)[::-1]
    
    print(f"{'Слой':<10} | {'Сила отклонения':<20} | {'Что происходит'}")
    print("-" * 50)
    
    for rank, layer_idx in enumerate(top_layers_indices[:5]):
        val = max_deviations_per_layer[layer_idx]
        print(f"Layer {layer_idx:<8} | {val:.2f}              | ⚠️ Критическая точка")
        
        if rank == 0:
            print(f"             |                    | 👑 СЮДА ПЕРВЫМ ДЕЛОМ СЕЙ СЕМЯ!")

    print("\n💡 ИНТЕРПРЕТАЦИЯ:")
    print("Красные линии на графике — это 'силы', которые уводят сеть с правильного пути.")
    print("Чем длиннее красная линия на конкретном слое, тем сильнее там влияние Хаоса.")
    print("Твои 'Мастера' должны компенсировать именно эти вектора (красные линии).")

if __name__ == "__main__":
    main()
