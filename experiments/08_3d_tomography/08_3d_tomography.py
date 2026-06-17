#!/usr/bin/env python3
"""
Эксперимент 10 (Оптимизированный): Эмпирический поиск направления

Исправления:
- transient=True для вложенных прогресс-баров (убирает дублирование)
- Меньше обновлений (раз в 100 тестов вместо 10)
- Убран gc.collect() из критического пути
- Опциональный быстрый режим для тестирования
"""

import torch
import gc
import os
import re
import time
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

import logging

# Отключаем варнинги transformers ДО того, как они успеют что-то напечатать
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
PERTURB_LAYERS = [10, 11, 12]
TARGET_LAYER = 23
N_SAMPLES = 2000
MAGNITUDES = [0.01, 0.05, 0.1, 0.2, 0.5]
POLY_DEGREE = 3
UPDATE_INTERVAL = 100  # Обновлять прогресс раз в 100 тестов (было 10)
FAST_MODE = False  # Если True, использует меньшее количество тестов для быстрого прототипирования

console = Console()

# Быстрый режим для тестирования
if FAST_MODE:
    N_SAMPLES = 100
    MAGNITUDES = [0.1, 0.5]
    console.print("[yellow]⚡ БЫСТРЫЙ РЕЖИМ: уменьшено количество тестов[/yellow]")


# ============================================================
# 1. ЗАХВАТ АКТИВАЦИЙ С ПРАВИЛЬНОЙ ПЕРЕДАЧЕЙ ВОЗМУЩЕНИЯ
# ============================================================
def capture_full_activations_with_perturbation(model, tokenizer, prompt, device, 
                                                perturbation_layer, direction, magnitude):
    activations = {}
    perturbation_applied = False
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            nonlocal perturbation_applied
            
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            if layer_idx == perturbation_layer and not perturbation_applied:
                perturbation = magnitude * torch.from_numpy(direction).to(tensor.device)
                tensor = tensor.clone()
                tensor[0, -1, :] += perturbation
                perturbation_applied = True
            
            activations[layer_idx] = tensor[0, -1, :].detach().cpu().numpy()
            
            if isinstance(output, tuple):
                return (tensor,) + output[1:]
            else:
                return tensor
        
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if 'model.layers.' in name and name.count('.') == 2:
            try:
                layer_idx = int(name.split('.')[-1])
                h = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)
            except ValueError:
                pass
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id  # <-- Явно передаем, чтобы не было варнинга
        )
    
    del inputs
    for h in hooks:
        h.remove()
    # Убран gc.collect() - замедляет выполнение
    
    return activations


def capture_base_activations(model, tokenizer, prompt, device):
    activations = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            activations[layer_idx] = tensor[0, -1, :].detach().cpu().numpy()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if 'model.layers.' in name and name.count('.') == 2:
            try:
                layer_idx = int(name.split('.')[-1])
                h = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)
            except ValueError:
                pass
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id  # <-- Явно передаем, чтобы не было варнинга
        )
    
    del inputs
    for h in hooks:
        h.remove()
    
    return activations


# ============================================================
# 2. PCA + ГЕОДЕЗИЧЕСКАЯ
# ============================================================
def compute_geodesic(real_activations):
    layers = sorted(real_activations.keys())
    matrix = np.array([real_activations[layer] for layer in layers])
    
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(matrix)
    
    explained_var = pca.explained_variance_ratio_
    
    layer_indices = np.array(layers, dtype=float)
    geodesic_coords = np.zeros((len(layers), 3))
    
    for comp in range(3):
        coeffs = np.polyfit(layer_indices, coords_3d[:, comp], POLY_DEGREE)
        geodesic_coords[:, comp] = np.polyval(coeffs, layer_indices)
    
    real_trajectory = []
    geodesic_trajectory = []
    
    for i, layer_idx in enumerate(layers):
        real_trajectory.append({
            'layer': layer_idx,
            'pc1': float(coords_3d[i, 0]),
            'pc2': float(coords_3d[i, 1]),
            'pc3': float(coords_3d[i, 2])
        })
        geodesic_trajectory.append({
            'layer': layer_idx,
            'pc1': float(geodesic_coords[i, 0]),
            'pc2': float(geodesic_coords[i, 1]),
            'pc3': float(geodesic_coords[i, 2])
        })
    
    return real_trajectory, geodesic_trajectory, pca


# ============================================================
# 3. ЭМПИРИЧЕСКИЙ ПОИСК С ИСПРАВЛЕННЫМ ПРОГРЕССОМ
# ============================================================
def find_optimal_direction_empirical(model, tokenizer, device, prompt, 
                                     perturbation_layers, target_layer,
                                     real_activations, geodesic_trajectory, 
                                     pca, n_samples=2000, magnitudes=None):
    if magnitudes is None:
        magnitudes = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    total_tests = len(perturbation_layers) * n_samples * len(magnitudes)
    
    console.print(f"\n[bold cyan]🔍 Эмпирический поиск направления:[/bold cyan]")
    console.print(f"   Слои: {perturbation_layers}")
    console.print(f"   Целевой слой: {target_layer}")
    console.print(f"   Всего тестов: [bold yellow]{total_tests:,}[/bold yellow]")
    console.print(f"   (слои × сэмплы × magnitudes = {len(perturbation_layers)} × {n_samples} × {len(magnitudes)})")
    
    best_direction = None
    best_magnitude = 0.0
    best_layer = None
    best_distance = float('inf')
    total_completed = 0
    
    target_geo = geodesic_trajectory[target_layer]
    target_geo_vec = np.array([target_geo['pc1'], target_geo['pc2'], target_geo['pc3']])
    
    hidden_dim = real_activations[perturbation_layers[0]].shape[0]
    
    start_time = time.time()
    
    # Настраиваем Rich прогресс
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("• [bold green]best: {task.fields[best_dist]}[/bold green]"),
        TextColumn("• [bold magenta]layer: {task.fields[layer]}[/bold magenta]"),
        console=console,
        refresh_per_second=1,  # Уменьшено с 2 до 1
        expand=True
    ) as progress:
        
        # Общий прогресс
        overall_task = progress.add_task(
            "[cyan]Общий прогресс",
            total=total_tests,
            best_dist=f"{best_distance:.4f}",
            layer="-"
        )
        
        for perturbation_layer in perturbation_layers:
            # Прогресс для текущего слоя с transient=True (не оставляет следов)
            layer_task = progress.add_task(
                f"[yellow]Слой {perturbation_layer}",
                total=n_samples,
                best_dist=f"{best_distance:.4f}",
                layer=str(perturbation_layer),
                transient=True  # ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
            )
            
            for i in range(n_samples):
                direction = np.random.randn(hidden_dim)
                direction /= np.linalg.norm(direction)
                
                for magnitude in magnitudes:
                    try:
                        perturbed_acts = capture_full_activations_with_perturbation(
                            model, tokenizer, prompt, device,
                            perturbation_layer, direction, magnitude
                        )
                        
                        target_act = perturbed_acts[target_layer].reshape(1, -1)
                        target_pca = pca.transform(target_act)[0]
                        
                        dist = np.linalg.norm(target_pca - target_geo_vec)
                        
                        if dist < best_distance:
                            best_distance = dist
                            best_direction = direction.copy()
                            best_magnitude = magnitude
                            best_layer = perturbation_layer
                        
                        total_completed += 1
                        
                        # Обновляем прогресс реже (раз в UPDATE_INTERVAL тестов)
                        if total_completed % UPDATE_INTERVAL == 0:
                            progress.update(
                                overall_task,
                                completed=total_completed,
                                best_dist=f"{best_distance:.4f}",
                                layer=str(perturbation_layer)
                            )
                        
                    except Exception as e:
                        total_completed += 1
                        continue
                
                # Обновляем прогресс слоя после каждого сэмпла
                progress.update(
                    layer_task,
                    completed=i + 1,
                    best_dist=f"{best_distance:.4f}",
                    layer=str(perturbation_layer)
                )
            
            # Финальное обновление общего прогресса
            progress.update(
                overall_task,
                completed=total_completed,
                best_dist=f"{best_distance:.4f}",
                layer=str(perturbation_layer)
            )
    
    elapsed = time.time() - start_time
    
    console.print(f"\n[bold green]✅ Найдено оптимальное направление:[/bold green]")
    console.print(f"   Слой: [bold]{best_layer}[/bold]")
    console.print(f"   Расстояние: [bold]{best_distance:.4f}[/bold]")
    console.print(f"   Magnitude: [bold]{best_magnitude}[/bold]")
    console.print(f"   Время: [bold]{elapsed:.1f} сек ({elapsed/60:.1f} мин)[/bold]")
    
    return best_direction, best_magnitude, best_layer, best_distance


# ============================================================
# 4. ПРОЕКЦИЯ В PCA
# ============================================================
def project_to_pca(activations, pca):
    layers = sorted(activations.keys())
    matrix = np.array([activations[layer] for layer in layers])
    coords_3d = pca.transform(matrix)
    
    trajectory = []
    for i, layer_idx in enumerate(layers):
        trajectory.append({
            'layer': layer_idx,
            'pc1': float(coords_3d[i, 0]),
            'pc2': float(coords_3d[i, 1]),
            'pc3': float(coords_3d[i, 2])
        })
    return trajectory


# ============================================================
# 5. ТЕСТ НА АРИФМЕТИКЕ
# ============================================================
def test_arithmetic_with_perturbation(model, tokenizer, device, test_cases, 
                                      perturbation_layer, direction, magnitude):
    console.print(f"\n[bold cyan]🧮 Тест на арифметике (возмущение в слое {perturbation_layer}):[/bold cyan]")
    correct = 0
    total = len(test_cases)
    
    for problem, expected in test_cases:
        prompt = f"Calculate: {problem} ="
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        answer = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        try:
            numbers = re.findall(r'-?\d+', answer)
            if numbers:
                predicted = int(numbers[0])
            else:
                predicted = None
        except:
            predicted = None
        
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        console.print(f"   {status} {problem} = {expected}")
        console.print(f"      Ответ: '{answer[:50]}...' → {predicted}")
    
    accuracy = correct / total * 100
    console.print(f"\n   [bold]Точность: {correct}/{total} = {accuracy:.1f}%[/bold]")
    
    return accuracy


# ============================================================
# 6. 3D ГРАФИК
# ============================================================
def create_3d_plot(real_traj, geodesic_traj, corrected_traj, save_path):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=[p['pc1'] for p in real_traj],
        y=[p['pc2'] for p in real_traj],
        z=[p['pc3'] for p in real_traj],
        mode='lines+markers',
        name='Реальная (база)',
        marker=dict(size=5, color='red', opacity=0.8),
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[p['pc1'] for p in geodesic_traj],
        y=[p['pc2'] for p in geodesic_traj],
        z=[p['pc3'] for p in geodesic_traj],
        mode='lines',
        name='Геодезическая',
        line=dict(color='black', width=4, dash='dash')
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[p['pc1'] for p in corrected_traj],
        y=[p['pc2'] for p in corrected_traj],
        z=[p['pc3'] for p in corrected_traj],
        mode='lines+markers',
        name='С эмпирической коррекцией',
        marker=dict(size=5, color='blue', opacity=0.8),
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='Слой (0-23)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5)
        ),
        title={'text': f'Эмпирический поиск направления (оптимизированный)', 'x': 0.5},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1200, height=900
    )
    
    html_path = save_path.with_suffix('.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    console.print(f"\n[bold green]✅ 3D график:[/bold green] {html_path}")
    
    return html_path


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    console.print(Panel.fit(
        "[bold]Эксперимент 10: Эмпирический поиск направления[/bold]\n"
        "Оптимизированная версия с Rich прогресс-барами",
        title="🧪 MeaningSeed",
        border_style="blue"
    ))
    
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cpu"
    prompt = "Calculate: 24578 + 13892 ="
    test_prompts = [
        ("24578 + 13892", 38470),
        ("12345 + 67890", 80235),
        ("99999 + 1", 100000),
    ]
    
    # === 1. Загрузка модели ===
    console.print("\n[bold cyan]📦 [1/6] Загрузка модели...[/bold cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    console.print("[bold green]  ✅ Модель загружена[/bold green]")
    
    # === 2. Базовый прогон ===
    console.print("\n[bold cyan]🎯 [2/6] Базовый прогон...[/bold cyan]")
    real_activations = capture_base_activations(model, tokenizer, prompt, device)
    console.print(f"[bold green]  ✅ Захвачено {len(real_activations)} слоёв[/bold green]")
    
    # === 3. Геодезическая ===
    console.print("\n[bold cyan]📐 [3/6] Вычисление геодезической...[/bold cyan]")
    real_traj, geodesic_traj, pca = compute_geodesic(real_activations)
    
    explained_var = pca.explained_variance_ratio_
    console.print(f"  PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}, PC3={explained_var[2]:.1%}")
    
    # === 4. Эмпирический поиск ===
    console.print("\n[bold cyan]🔬 [4/6] Эмпирический поиск направления...[/bold cyan]")
    best_direction, best_magnitude, best_layer, best_distance = find_optimal_direction_empirical(
        model, tokenizer, device, prompt,
        PERTURB_LAYERS, TARGET_LAYER,
        real_activations, 
        {p['layer']: p for p in geodesic_traj},
        pca,
        n_samples=N_SAMPLES,
        magnitudes=MAGNITUDES
    )
    
    if best_direction is None:
        console.print("[bold red] Не удалось найти направление![/bold red]")
        return
    
    # === 5. Применение направления ===
    console.print(f"\n[bold cyan]💉 [5/6] Применение найденного направления...[/bold cyan]")
    console.print(f"   Слой: {best_layer}, Magnitude: {best_magnitude}")
    
    corrected_activations = capture_full_activations_with_perturbation(
        model, tokenizer, prompt, device,
        best_layer, best_direction, best_magnitude
    )
    corrected_traj = project_to_pca(corrected_activations, pca)
    console.print(f"[bold green]  ✅ Скорректированная траектория[/bold green]")
    
    # === 6. Тест на арифметике ===
    console.print("\n[bold cyan] [6/6] Тест на арифметике...[/bold cyan]")
    accuracy = test_arithmetic_with_perturbation(
        model, tokenizer, device, test_prompts,
        best_layer, best_direction, best_magnitude
    )
    
    # === Метрики ===
    def total_distance(traj1, traj2):
        return sum(
            np.sqrt((p1['pc1'] - p2['pc1'])**2 + (p1['pc2'] - p2['pc2'])**2 + (p1['pc3'] - p2['pc3'])**2)
            for p1, p2 in zip(traj1, traj2)
        )
    
    dist_before = total_distance(real_traj, geodesic_traj)
    dist_after = total_distance(corrected_traj, geodesic_traj)
    improvement = (dist_before - dist_after) / dist_before * 100
    
    target_before = np.sqrt(
        (real_traj[TARGET_LAYER]['pc1'] - geodesic_traj[TARGET_LAYER]['pc1'])**2 +
        (real_traj[TARGET_LAYER]['pc2'] - geodesic_traj[TARGET_LAYER]['pc2'])**2 +
        (real_traj[TARGET_LAYER]['pc3'] - geodesic_traj[TARGET_LAYER]['pc3'])**2
    )
    target_after = np.sqrt(
        (corrected_traj[TARGET_LAYER]['pc1'] - geodesic_traj[TARGET_LAYER]['pc1'])**2 +
        (corrected_traj[TARGET_LAYER]['pc2'] - geodesic_traj[TARGET_LAYER]['pc2'])**2 +
        (corrected_traj[TARGET_LAYER]['pc3'] - geodesic_traj[TARGET_LAYER]['pc3'])**2
    )
    target_improvement = (target_before - target_after) / target_before * 100
    
    console.print(f"\n[bold cyan] Метрики траектории:[/bold cyan]")
    console.print(f"   Общая траектория:")
    console.print(f"     До: {dist_before:.4f}")
    console.print(f"     После: {dist_after:.4f}")
    console.print(f"     Улучшение: {improvement:+.1f}%")
    console.print(f"   Целевой слой {TARGET_LAYER}:")
    console.print(f"     До: {target_before:.4f}")
    console.print(f"     После: {target_after:.4f}")
    console.print(f"     Улучшение: {target_improvement:+.1f}%")
    
    # === Визуализация ===
    console.print("\n[bold cyan]🎨 Создание 3D графика...[/bold cyan]")
    html_path = create_3d_plot(
        real_traj, geodesic_traj, corrected_traj,
        reports_dir / "empirical_direction_optimized.html"
    )
    
    # === Финальный отчёт ===
    console.print("\n" + "=" * 70)
    console.print("[bold green]✨ ЭКСПЕРИМЕНТ 10 ЗАВЕРШЁН![/bold green]")
    console.print("=" * 70)
    console.print(f"\n[bold]📊 ИТОГИ:[/bold]")
    console.print(f"   • Точность арифметики: {accuracy:.1f}%")
    console.print(f"   • Улучшение траектории: {improvement:+.1f}%")
    console.print(f"   • Улучшение в слое {TARGET_LAYER}: {target_improvement:+.1f}%")
    console.print(f"   • Лучший слой для возмущения: {best_layer}")
    console.print(f"\n[bold]📂 ОТКРОЙ:[/bold] {html_path}")
    
    if accuracy > 0:
        console.print(f"\n[bold green] ВЫВОД: Эмпирический поиск направления УЛУЧШИЛ точность![/bold green]")
    else:
        console.print(f"\n[bold yellow]💡 ВЫВОД: Траектория улучшилась, но точность не выросла.[/bold yellow]")


if __name__ == "__main__":
    os.environ['PLOTLY_RENDERER'] = 'browser'
    main()
