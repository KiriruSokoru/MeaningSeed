#!/usr/bin/env python3
"""
================================================================================
Эксперимент 04: Time Reversal Challenge (ИСПРАВЛЕННАЯ ВЕРСИЯ)
================================================================================
"""

import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()


def load_sepmax_data(data_path: Path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_intermediate_target_prompts(sepmax_data, num_samples=20):
    import re
    prompts_a = sepmax_data['prompts_a'][:num_samples]
    prompts_b = sepmax_data['prompts_b'][:num_samples]
    proj_a = np.array(sepmax_data['projection_a'])[:num_samples]
    proj_b = np.array(sepmax_data['projection_b'])[:num_samples]
    
    time_reversal_prompts = []
    
    for i in range(num_samples):
        original_a = prompts_a[i]
        original_b = prompts_b[i]
        
        match_a = re.search(r'Compute (\d+) - (\d+)', original_a)
        if match_a:
            x_a, y_a = int(match_a.group(1)), int(match_a.group(2))
            answer_a = x_a - y_a
            intermediate = answer_a / 2
            prompt = f"Compute the halfway point: {x_a} - {y_a} = ? Half of that is: {intermediate}"
            time_reversal_prompts.append({
                'prompt': prompt,
                'type': 'time_reversal',
                'original_a': original_a,
                'original_b': original_b,
                'expected_activation_ratio': 0.5,
                'intermediate_value': float(intermediate),
                'full_answer_a': int(answer_a)
            })
    
    return time_reversal_prompts


def collect_activations_for_prompts(model, tokenizer, prompts, device, description="Сбор активаций"):
    activations = []
    prompt_list = []
    
    with Progress(
        TextColumn(f"[cyan]{description}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("", total=len(prompts))
        
        for i, prompt_item in enumerate(prompts):
            if isinstance(prompt_item, dict):
                prompt = prompt_item['prompt']
            else:
                prompt = prompt_item
            
            progress.update(task, description=f"[cyan]{description} [{i+1}/{len(prompts)}] {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            last_hidden = outputs.hidden_states[-1]
            avg_hidden = last_hidden.mean(dim=1).squeeze().cpu().numpy()
            
            activations.append(avg_hidden)
            prompt_list.append({'prompt': prompt})
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress.update(task, advance=1)
    
    return np.array(activations), prompt_list


def project_to_sepmax(activations, sepmax_vectors):
    sepmax_vectors = np.array(sepmax_vectors)
    return activations @ sepmax_vectors.T


def compute_flow_violation_angle(proj_trajectory, proj_natural_path, proj_time_reversed_path):
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)
    
    if len(proj_trajectory) > 1:
        test_vector = proj_trajectory[-1] - proj_trajectory[0]
    else:
        test_vector = proj_trajectory[0]
    
    if len(proj_natural_path) > 1:
        natural_vector = proj_natural_path[-1] - proj_natural_path[0]
    else:
        natural_vector = proj_natural_path[0]
    
    if len(proj_time_reversed_path) > 1:
        reversed_vector = proj_time_reversed_path[-1] - proj_time_reversed_path[0]
    else:
        reversed_vector = proj_time_reversed_path[0]
    
    test_norm = normalize(test_vector)
    natural_norm = normalize(natural_vector)
    reversed_norm = normalize(reversed_vector)
    
    dot_with_natural = np.clip(np.dot(test_norm, natural_norm), -1, 1)
    dot_with_reversed = np.clip(np.dot(test_norm, reversed_norm), -1, 1)
    
    angle_with_natural = np.arccos(dot_with_natural) * 180 / np.pi
    angle_with_reversed = np.arccos(dot_with_reversed) * 180 / np.pi
    
    return {
        'angle_with_natural_deg': float(angle_with_natural),
        'angle_with_reversed_deg': float(angle_with_reversed),
        'similarity_to_natural': float((dot_with_natural + 1) / 2),
        'similarity_to_reversed': float((dot_with_reversed + 1) / 2),
        'prefers_natural': bool(angle_with_natural < angle_with_reversed)
    }


def visualize_time_reversal(proj_natural, proj_reversed, proj_test, save_path, metrics):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if len(proj_natural) > 0:
        ax.plot(proj_natural[:, 0], proj_natural[:, 1], 'b-', linewidth=2, alpha=0.7, label='Естественный путь (A→B)')
        ax.scatter(proj_natural[0, 0], proj_natural[0, 1], c='darkblue', s=100, marker='o', label='Старт A')
        ax.scatter(proj_natural[-1, 0], proj_natural[-1, 1], c='blue', s=100, marker='s', label='Цель B')
    
    if len(proj_reversed) > 0:
        ax.plot(proj_reversed[:, 0], proj_reversed[:, 1], 'r-', linewidth=2, alpha=0.7, label='Time-reversed путь (B→A)')
        ax.scatter(proj_reversed[0, 0], proj_reversed[0, 1], c='darkred', s=100, marker='o', label='Старт B')
        ax.scatter(proj_reversed[-1, 0], proj_reversed[-1, 1], c='red', s=100, marker='s', label='Цель A')
    
    if len(proj_test) > 0:
        ax.plot(proj_test[:, 0], proj_test[:, 1], 'g--', linewidth=2, alpha=0.9, label='Time Reversal Challenge')
        ax.scatter(proj_test[0, 0], proj_test[0, 1], c='darkgreen', s=100, marker='o', label='Старт (из B)')
        
        if len(proj_natural) > 0:
            midpoint = (proj_natural[0] + proj_natural[-1]) / 2
            ax.scatter(midpoint[0], midpoint[1], c='black', s=150, marker='*', 
                      edgecolors='yellow', linewidth=2, label='Промежуточная цель (IT)')
    
    ax.set_xlabel('SepMax Component 1', fontsize=12)
    ax.set_ylabel('SepMax Component 2', fontsize=12)
    ax.set_title('Time Reversal Challenge', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    text = f"Угол с естественным путем: {metrics.get('angle_with_natural_deg', 0):.1f}°\n"
    text += f"Угол с time-reversed путем: {metrics.get('angle_with_reversed_deg', 0):.1f}°\n"
    text += f"Предпочитает естественный путь: {metrics.get('prefers_natural', False)}"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    console.print(Panel.fit(
        "[bold cyan]Эксперимент 04: Time Reversal Challenge[/bold cyan]",
        border_style="cyan"
    ))
    
    # Загрузка данных
    console.print("\n[bold yellow]1. Загрузка SepMax данных[/bold yellow]")
    data_path = Path(__file__).parent / "activations" / "sepmax_data.json"
    if not data_path.exists():
        console.print("[red]❌ Файл sepmax_data.json не найден![/red]")
        return
    
    sepmax_data = load_sepmax_data(data_path)
    console.print(f"[green]✓ Загружены данные, d' = {sepmax_data['metrics']['d_prime_euclidean']:.2f}[/green]")
    
    # Создание промптов
    console.print("\n[bold yellow]2. Создание промптов для Time Reversal[/bold yellow]")
    time_reversal_prompts = create_intermediate_target_prompts(sepmax_data, num_samples=20)
    console.print(f"[green]✓ Создано {len(time_reversal_prompts)} промптов[/green]")
    
    if len(time_reversal_prompts) == 0:
        console.print("[red]Нет промптов для time reversal challenge[/red]")
        return
    
    # Загрузка модели
    console.print("\n[bold yellow]3. Загрузка модели[/bold yellow]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.print("[green]✓ Модель загружена[/green]")
    
    # Сбор активаций
    console.print("\n[bold yellow]4. Сбор активаций для Time Reversal[/bold yellow]")
    test_activations, _ = collect_activations_for_prompts(
        model, tokenizer, time_reversal_prompts, device, "Сбор TR активаций"
    )
    
    # Проекция в SepMax
    sepmax_vectors = np.array(sepmax_data['sepmax_vectors'])
    proj_test = project_to_sepmax(test_activations, sepmax_vectors)
    proj_natural = np.array(sepmax_data['projection_a'])
    proj_reversed = np.array(sepmax_data['projection_b'])
    
    # Вычисление метрик
    test_metrics = []
    for i in range(min(len(proj_test), len(proj_natural), len(proj_reversed))):
        metrics = compute_flow_violation_angle(
            proj_test[i:i+1],
            proj_natural[i:i+1],
            proj_reversed[i:i+1]
        )
        test_metrics.append(metrics)
    
    angles_natural = [m['angle_with_natural_deg'] for m in test_metrics]
    angles_reversed = [m['angle_with_reversed_deg'] for m in test_metrics]
    prefers_natural = [m['prefers_natural'] for m in test_metrics]
    
    # Вывод результатов
    console.print("\n[bold yellow]5. Результаты[/bold yellow]")
    table = Table(title="Оценка отклонения от естественного потока")
    table.add_column("Метрика", style="cyan")
    table.add_column("Среднее", style="green")
    table.add_column("Стд. откл.", style="dim")
    
    table.add_row("Угол с естественным путём (A→B)", f"{np.mean(angles_natural):.1f}°", f"{np.std(angles_natural):.1f}°")
    table.add_row("Угол с time-reversed путём (B→A)", f"{np.mean(angles_reversed):.1f}°", f"{np.std(angles_reversed):.1f}°")
    table.add_row("Доля предпочитающих естественный путь", f"{np.mean(prefers_natural)*100:.1f}%", "")
    
    console.print(table)
    
    if np.mean(angles_natural) < np.mean(angles_reversed):
        console.print("\n[bold green]✅ Модель предпочитает естественный путь (A→B)[/bold green]")
    else:
        console.print("\n[bold yellow]⚠️ Модель не показывает явного предпочтения[/bold yellow]")
    
    # Визуализация
    console.print("\n[bold yellow]6. Визуализация[/bold yellow]")
    viz_path = Path(__file__).parent / "visualizations" / "time_reversal_trajectories.png"
    natural_path = np.array([proj_natural[0], proj_natural[-1]]) if len(proj_natural) > 0 else np.array([[0,0], [0,0]])
    reversed_path = np.array([proj_reversed[0], proj_reversed[-1]]) if len(proj_reversed) > 0 else np.array([[0,0], [0,0]])
    
    visualize_time_reversal(
        natural_path, reversed_path, proj_test[:1], viz_path,
        {'angle_with_natural_deg': np.mean(angles_natural),
         'angle_with_reversed_deg': np.mean(angles_reversed),
         'prefers_natural': np.mean(prefers_natural) > 0.5}
    )
    console.print(f"[green]✓ Визуализация: {viz_path}[/green]")
    
    # Сохранение результатов
    console.print("\n[bold yellow]7. Сохранение результатов[/bold yellow]")
    
    results = {
        'summary': {
            'mean_angle_with_natural': float(np.mean(angles_natural)),
            'std_angle_with_natural': float(np.std(angles_natural)),
            'mean_angle_with_reversed': float(np.mean(angles_reversed)),
            'std_angle_with_reversed': float(np.std(angles_reversed)),
            'prefers_natural_ratio': float(np.mean(prefers_natural)),
            'num_samples': len(test_metrics)
        }
    }
    
    results_path = Path(__file__).parent / "results" / "time_reversal_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Результаты сохранены: {results_path}[/green]")
    
    # Очистка
    del model
    torch.cuda.empty_cache()
    console.print("\n[bold green]✅ Time Reversal Challenge завершён![/bold green]")


if __name__ == "__main__":
    main()
