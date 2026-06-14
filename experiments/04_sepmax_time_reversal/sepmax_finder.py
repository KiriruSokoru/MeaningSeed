#!/usr/bin/env python3
"""
================================================================================
Эксперимент 04: Поиск SepMax проекции
================================================================================

Цель: Найти 2D-проекцию нейронного пространства, в которой траектории A→B и B→A
максимально разделены (аналог SepMax из Nature Neuroscience 2025).

Методология:
1. Собираем активации скрытых слоёв для двух типов промптов:
   - Тип A: "Compute X + Y" (A→B)
   - Тип B: "Compute Y + X" (B→A) - тот же ответ, но другой порядок

2. Приводим активации к единой размерности (усреднение по токенам)

3. Находим 2D-проекцию, максимизирующую разделение:
   - Первая компонента: вектор между центрами A и B
   - Вторая компонента: направление максимальной дисперсии, ортогональное первой

4. Вычисляем d' (discriminability index) — меру разделимости

5. Визуализируем проекцию и сохраняем данные для time reversal challenge

Выходные файлы:
- visualizations/sepmax_projection.png — график проекции
- activations/sepmax_data.json — данные для следующего шага
================================================================================
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()


def load_arithmetic_tasks(tasks_path: Path, limit: int = 50):
    """
    Загружает арифметические задачи и создаёт пары A→B и B→A.
    
    Args:
        tasks_path: путь к файлу tasks/arithmetic_5digit_fixed.json
        limit: количество задач для использования
    
    Returns:
        prompts_a: список промптов для A→B (исходный порядок)
        prompts_b: список промптов для B→A (обратный порядок)
        answers: список правильных ответов
    """
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    
    prompts_a = []
    prompts_b = []
    answers = []
    
    for task in all_tasks[:limit]:
        expr = task['original_prompt'].replace("Calculate:", "").strip()
        correct_answer = task['answer']
        
        # Парсим выражение
        if ' + ' in expr:
            a, rest = expr.split(' + ')
            b, _ = rest.split(' =')
            prompts_a.append(f"Compute {a} + {b}")
            prompts_b.append(f"Compute {b} + {a}")
            answers.append(correct_answer)
        elif ' - ' in expr:
            a, rest = expr.split(' - ')
            b, _ = rest.split(' =')
            prompts_a.append(f"Compute {a} - {b}")
            prompts_b.append(f"Compute {b} - {a}")
            answers.append(correct_answer)
    
    return prompts_a, prompts_b, answers


def collect_activations(model, tokenizer, prompts, device, description="Сбор активаций"):
    """
    Собирает активации последнего скрытого слоя для каждого промпта.
    
    Args:
        model: загруженная модель
        tokenizer: токенизатор
        prompts: список промптов
        device: устройство (cuda/cpu)
        description: описание для прогресс-бара
    
    Returns:
        activations: numpy array формы (len(prompts), hidden_dim)
        где hidden_dim - размерность скрытого состояния
    """
    activations = []
    
    # Настраиваем прогресс-бар
    with Progress(
        TextColumn(f"[cyan]{description}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("", total=len(prompts))
        
        for i, prompt in enumerate(prompts):
            # Обновляем описание с текущим промптом
            progress.update(task, description=f"[cyan]{description} [{i+1}/{len(prompts)}] {prompt[:50]}...")
            
            # Токенизируем
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Прямой проход с захватом скрытых состояний
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Берём последний скрытый слой
            # outputs.hidden_states — это tuple из (num_layers + 1) тензоров
            # Индекс -1 — последний слой, форма [batch, seq_len, hidden_dim]
            last_hidden = outputs.hidden_states[-1]
            
            # Усредняем по токенам (по оси seq_len)
            # Получаем [batch, hidden_dim], batch у нас 1
            avg_hidden = last_hidden.mean(dim=1).squeeze().cpu().numpy()
            
            activations.append(avg_hidden)
            
            # Очищаем память
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress.update(task, advance=1)
    
    return np.array(activations)


def find_sepmax_projection(activations_a, activations_b, n_components=2):
    """
    Находит SepMax проекцию — 2D-подпространство, максимизирующее разделение
    между траекториями A→B и B→A.
    
    Алгоритм:
    1. Вычисляем центры масс для A и B
    2. Первая компонента: единичный вектор от центра B к центру A
    3. Вторая компонента: направление максимальной дисперсии в объединённых данных,
       ортогональное первой компоненте (через PCA с последующей ортогонализацией)
    
    Args:
        activations_a: np.array [n_trials_a, hidden_dim] — активации для A→B
        activations_b: np.array [n_trials_b, hidden_dim] — активации для B→A
        n_components: размерность проекции (обычно 2)
    
    Returns:
        sepmax_vectors: np.array [n_components, hidden_dim] — проекционные векторы
        proj_a: np.array [n_trials_a, n_components] — проекции A
        proj_b: np.array [n_trials_b, n_components] — проекции B
        metrics: dict с метриками (center_a, center_b, d_prime)
    """
    # Центры масс
    center_a = activations_a.mean(axis=0)
    center_b = activations_b.mean(axis=0)
    
    # Первая компонента: вектор между центрами (направление разделения)
    sep_vector = center_a - center_b
    sep_vector = sep_vector / (np.linalg.norm(sep_vector) + 1e-8)
    
    # Объединяем данные для поиска второй компоненты
    all_data = np.vstack([activations_a, activations_b])
    all_centered = all_data - all_data.mean(axis=0)
    
    # PCA для поиска направлений максимальной дисперсии
    pca = PCA(n_components=min(n_components, all_centered.shape[1]))
    pca.fit(all_centered)
    
    # Выбираем компоненту, максимально ортогональную к sep_vector
    # (наименьшее абсолютное значение скалярного произведения)
    best_corr = float('inf')
    best_idx = 0
    
    for i in range(pca.components_.shape[0]):
        corr = abs(np.dot(pca.components_[i], sep_vector))
        if corr < best_corr:
            best_corr = corr
            best_idx = i
    
    orthogonal_vector = pca.components_[best_idx]
    # Ортогонализуем относительно sep_vector (на всякий случай)
    orthogonal_vector = orthogonal_vector - np.dot(orthogonal_vector, sep_vector) * sep_vector
    orthogonal_vector = orthogonal_vector / (np.linalg.norm(orthogonal_vector) + 1e-8)
    
    # Формируем матрицу проекции
    sepmax_vectors = np.vstack([sep_vector, orthogonal_vector])
    
    # Проецируем данные
    proj_a = activations_a @ sepmax_vectors.T
    proj_b = activations_b @ sepmax_vectors.T
    
    # Вычисляем d' (discriminability index)
    # d' = (μ1 - μ2) / sqrt((σ1² + σ2²)/2)
    center_a_proj = proj_a.mean(axis=0)
    center_b_proj = proj_b.mean(axis=0)
    var_a_proj = proj_a.var(axis=0)
    var_b_proj = proj_b.var(axis=0)
    
    # d' по первой компоненте (направлению разделения)
    d_prime_1 = abs(center_a_proj[0] - center_b_proj[0]) / np.sqrt((var_a_proj[0] + var_b_proj[0]) / 2)
    
    # d' по второй компоненте
    d_prime_2 = abs(center_a_proj[1] - center_b_proj[1]) / np.sqrt((var_a_proj[1] + var_b_proj[1]) / 2)
    
    # Евклидово расстояние в 2D
    d_prime_euclidean = np.linalg.norm(center_a_proj - center_b_proj) / np.sqrt((var_a_proj + var_b_proj).mean())
    
    metrics = {
        'center_a': center_a_proj.tolist(),
        'center_b': center_b_proj.tolist(),
        'var_a': var_a_proj.tolist(),
        'var_b': var_b_proj.tolist(),
        'd_prime_1': float(d_prime_1),
        'd_prime_2': float(d_prime_2),
        'd_prime_euclidean': float(d_prime_euclidean),
        'orthogonality_corr': float(best_corr),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()
    }
    
    return sepmax_vectors, proj_a, proj_b, metrics


def visualize_projection(proj_a, proj_b, metrics, save_path):
    """
    Визуализирует SepMax проекцию.
    
    Создаёт график с:
    - Точками A→B (синие) и B→A (красные)
    - Центрами масс (звёзды)
    - Эллипсами (1 стандартное отклонение)
    - Значением d' на графике
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Точки данных
    ax.scatter(proj_a[:, 0], proj_a[:, 1], c='blue', alpha=0.6, s=60, label='A→B (исходный порядок)')
    ax.scatter(proj_b[:, 0], proj_b[:, 1], c='red', alpha=0.6, s=60, label='B→A (обратный порядок)')
    
    # Центры масс
    center_a = np.array(metrics['center_a'])
    center_b = np.array(metrics['center_b'])
    ax.scatter(center_a[0], center_a[1], c='darkblue', marker='*', s=300, edgecolors='black', linewidth=1.5, label='Центр A→B')
    ax.scatter(center_b[0], center_b[1], c='darkred', marker='*', s=300, edgecolors='black', linewidth=1.5, label='Центр B→A')
    
    # Эллипсы (1 std)
    from matplotlib.patches import Ellipse
    
    var_a = np.array(metrics['var_a'])
    var_b = np.array(metrics['var_b'])
    
    ellipse_a = Ellipse(xy=center_a, width=2*np.sqrt(var_a[0]), height=2*np.sqrt(var_a[1]), 
                        angle=0, alpha=0.3, color='blue', label='A→B (1σ)')
    ellipse_b = Ellipse(xy=center_b, width=2*np.sqrt(var_b[0]), height=2*np.sqrt(var_b[1]), 
                        angle=0, alpha=0.3, color='red', label='B→A (1σ)')
    ax.add_patch(ellipse_a)
    ax.add_patch(ellipse_b)
    
    # Линия между центрами
    ax.plot([center_a[0], center_b[0]], [center_a[1], center_b[1]], 'k--', alpha=0.5, linewidth=1)
    
    # Настройки графика
    ax.set_xlabel('SepMax Component 1 (направление разделения)', fontsize=12)
    ax.set_ylabel('SepMax Component 2 (максимальная дисперсия, ортогональная)', fontsize=12)
    ax.set_title('SepMax проекция: траектории A→B vs B→A', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Добавляем текст с метриками
    text = f"d'₁ = {metrics['d_prime_1']:.2f} (по компоненте 1)\n"
    text += f"d'₂ = {metrics['d_prime_2']:.2f} (по компоненте 2)\n"
    text += f"d' (Euclidean) = {metrics['d_prime_euclidean']:.2f}\n"
    text += f"Ортогональность компонент: {metrics['orthogonality_corr']:.4f}"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Визуализация сохранена: {save_path}[/green]")


def print_metrics_table(metrics):
    """Выводит метрики в виде Rich таблицы"""
    table = Table(title="Метрики разделимости SepMax")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="green")
    
    table.add_row("d'₁ (по компоненте 1)", f"{metrics['d_prime_1']:.4f}")
    table.add_row("d'₂ (по компоненте 2)", f"{metrics['d_prime_2']:.4f}")
    table.add_row("d' (Euclidean 2D)", f"{metrics['d_prime_euclidean']:.4f}")
    table.add_row("Ортогональность компонент", f"{metrics['orthogonality_corr']:.6f}")
    table.add_row("PCA explained variance (comp1)", f"{metrics['pca_explained_variance_ratio'][0]:.4f}")
    if len(metrics['pca_explained_variance_ratio']) > 1:
        table.add_row("PCA explained variance (comp2)", f"{metrics['pca_explained_variance_ratio'][1]:.4f}")
    
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold cyan]Эксперимент 04: Поиск SepMax проекции[/bold cyan]\n"
        "Аналог Nature Neuroscience 2025 — находим проекцию с максимальным разделением",
        border_style="cyan"
    ))
    
    # ==================== 1. ЗАГРУЗКА ЗАДАЧ ====================
    console.print("\n[bold yellow]1. Загрузка арифметических задач[/bold yellow]")
    
    tasks_path = Path("../02_arithmetic_masters/tasks/arithmetic_5digit_fixed.json")
    if not tasks_path.exists():
        # Пробуем альтернативный путь
        tasks_path = Path("/home/rillki/meaningseed_experiments/experiments/02_arithmetic_masters/tasks/arithmetic_5digit_fixed.json")
    
    if not tasks_path.exists():
        console.print("[red]❌ Файл с задачами не найден![/red]")
        console.print(f"[dim]Искали: {tasks_path}[/dim]")
        return
    
    prompts_a, prompts_b, answers = load_arithmetic_tasks(tasks_path, limit=50)
    
    console.print(f"[green]✓ Загружено {len(prompts_a)} пар задач[/green]")
    console.print(f"[dim]Пример A→B: {prompts_a[0]}[/dim]")
    console.print(f"[dim]Пример B→A: {prompts_b[0]}[/dim]")
    
    # ==================== 2. ЗАГРУЗКА МОДЕЛИ ====================
    console.print("\n[bold yellow]2. Загрузка модели Qwen Base[/bold yellow]")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Устройство: {device}[/dim]")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Получаем размерность скрытого состояния
    hidden_dim = model.config.hidden_size
    console.print(f"[green]✓ Модель загружена[/green]")
    console.print(f"[dim]Размерность скрытого состояния: {hidden_dim}[/dim]")
    
    # ==================== 3. СБОР АКТИВАЦИЙ ====================
    console.print("\n[bold yellow]3. Сбор активаций скрытых слоёв[/bold yellow]")
    
    activations_a = collect_activations(model, tokenizer, prompts_a, device, "Сбор A→B")
    activations_b = collect_activations(model, tokenizer, prompts_b, device, "Сбор B→A")
    
    console.print(f"[green]✓ Активации собраны[/green]")
    console.print(f"[dim]Форма A: {activations_a.shape}[/dim]")
    console.print(f"[dim]Форма B: {activations_b.shape}[/dim]")
    
    # ==================== 4. ПОИСК SEPMAX ПРОЕКЦИИ ====================
    console.print("\n[bold yellow]4. Поиск SepMax проекции[/bold yellow]")
    
    sepmax_vectors, proj_a, proj_b, metrics = find_sepmax_projection(activations_a, activations_b)
    
    console.print(f"[green]✓ SepMax проекция найдена[/green]")
    console.print(f"[dim]Размерность проекции: {sepmax_vectors.shape[0]} x {sepmax_vectors.shape[1]}[/dim]")
    
    # ==================== 5. ВЫВОД МЕТРИК ====================
    console.print("\n[bold yellow]5. Метрики разделимости[/bold yellow]")
    print_metrics_table(metrics)
    
    # Интерпретация d'
    d_prime = metrics['d_prime_euclidean']
    if d_prime < 0.5:
        interpretation = "слабая разделимость (почти неразличимы)"
    elif d_prime < 1.0:
        interpretation = "заметная разделимость"
    elif d_prime < 1.5:
        interpretation = "хорошая разделимость"
    else:
        interpretation = "очень хорошая разделимость"
    
    console.print(f"[cyan]Интерпретация d' = {d_prime:.2f}: {interpretation}[/cyan]")
    
    # ==================== 6. ВИЗУАЛИЗАЦИЯ ====================
    console.print("\n[bold yellow]6. Визуализация[/bold yellow]")
    
    viz_path = Path(__file__).parent / "visualizations" / "sepmax_projection.png"
    viz_path.parent.mkdir(exist_ok=True)
    
    visualize_projection(proj_a, proj_b, metrics, viz_path)
    
    # ==================== 7. СОХРАНЕНИЕ ДАННЫХ ====================
    console.print("\n[bold yellow]7. Сохранение данных для Time Reversal Challenge[/bold yellow]")
    
    # Конвертируем numpy в обычные float для JSON
    def to_serializable(arr):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, np.float32) or isinstance(arr, np.float64):
            return float(arr)
        if isinstance(arr, np.int32) or isinstance(arr, np.int64):
            return int(arr)
        return arr
    
    sepmax_data = {
        'sepmax_vectors': to_serializable(sepmax_vectors),
        'projection_a': to_serializable(proj_a),
        'projection_b': to_serializable(proj_b),
        'metrics': metrics,
        'hidden_dim': hidden_dim,
        'num_samples_a': len(activations_a),
        'num_samples_b': len(activations_b),
        'prompts_a': prompts_a,
        'prompts_b': prompts_b,
        'answers': to_serializable(answers)
    }
    
    data_path = Path(__file__).parent / "activations" / "sepmax_data.json"
    data_path.parent.mkdir(exist_ok=True)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(sepmax_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Данные сохранены: {data_path}[/green]")
    console.print(f"[dim]Размер файла: {data_path.stat().st_size / 1024:.1f} KB[/dim]")
    
    # ==================== 8. ОЧИСТКА ====================
    del model
    torch.cuda.empty_cache()
    
    console.print("\n[bold green]✅ SepMax проекция найдена и сохранена![/bold green]")
    console.print("\n[bold]Следующий шаг:[/bold] запусти time_reversal_challenge.py")


if __name__ == "__main__":
    from rich.panel import Panel
    main()
