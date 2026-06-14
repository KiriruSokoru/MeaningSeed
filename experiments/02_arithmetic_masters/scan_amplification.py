#!/usr/bin/env python3
"""
Сканирование коэффициентов усиления - с детальным прогрессом
"""

import json
import torch
import gc
import re
import random
import copy
import time
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import matplotlib.pyplot as plt

console = Console()


def load_tasks(tasks_path: Path, limit=50):
    with open(tasks_path) as f:
        tasks = json.load(f)
    return tasks[:limit]


def test_model_with_progress(model, tokenizer, tasks, device, description="Тестирование"):
    """Тестирует модель с детальным прогрессом по каждой задаче"""
    correct = 0
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]{description}", total=len(tasks))
        
        for i, task_data in enumerate(tasks):
            prompt = task_data['prompt']
            correct_answer = task_data['answer']
            
            # Показываем текущую задачу
            progress.update(task, description=f"[cyan]{description} [{i+1}/{len(tasks)}] {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            numbers = re.findall(r'-?\d+', response)
            if numbers:
                try:
                    predicted = int(numbers[-1])
                    if predicted == correct_answer:
                        correct += 1
                        progress.console.print(f"[dim]  ✓ {prompt} = {correct_answer}[/dim]")
                    else:
                        progress.console.print(f"[dim]  ✗ {prompt} = {correct_answer} (получено {predicted})[/dim]")
                except:
                    progress.console.print(f"[dim]  ✗ {prompt} = {correct_answer} (ошибка парсинга)[/dim]")
            else:
                progress.console.print(f"[dim]  ✗ {prompt} = {correct_answer} (нет числа)[/dim]")
            
            del inputs, outputs
            
            # Принудительно обновляем прогресс
            progress.update(task, advance=1)
    
    return correct / len(tasks) if tasks else 0


def get_master_neurons(seed_path: Path):
    with open(seed_path) as f:
        seed = json.load(f)
    return seed.get('masters', [])


def apply_master_seed(model, masters, amplification):
    modified = 0
    for m in masters:
        layer_name = m['layer']
        neuron_idx = m['neuron']
        
        module = model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                break
        
        if module is not None and hasattr(module, 'weight'):
            if neuron_idx < module.weight.shape[0]:
                module.weight.data[neuron_idx, :] *= amplification
                modified += 1
                if hasattr(module, 'bias') and module.bias is not None:
                    if neuron_idx < module.bias.shape[0]:
                        module.bias.data[neuron_idx] *= amplification
    return modified


def apply_random_seed(model, num_neurons=60, amplification=1.3):
    linear_layers = []
    for name, module in model.named_modules():
        if 'Linear' in str(module.__class__):
            linear_layers.append((name, module))
    
    modified = 0
    for _ in range(num_neurons):
        layer_name, module = random.choice(linear_layers)
        neuron_idx = random.randint(0, module.weight.shape[0] - 1)
        
        module.weight.data[neuron_idx, :] *= amplification
        modified += 1
        
        if hasattr(module, 'bias') and module.bias is not None:
            if neuron_idx < module.bias.shape[0]:
                module.bias.data[neuron_idx] *= amplification
    
    return modified


def main():
    console.print(Panel.fit(
        "[bold cyan]Сканирование коэффициентов усиления[/bold cyan]\n"
        "Мастер-нейроны vs случайные нейроны",
        border_style="cyan"
    ))
    
    base_dir = Path(__file__).parent
    seed_path = base_dir / "masters" / "arithmetic_seed.json"
    tasks_path = base_dir / "tasks" / "arithmetic_5digit_fixed.json"
    
    tasks = load_tasks(tasks_path, 50)
    masters = get_master_neurons(seed_path)
    
    console.print(f"[green]✓ Задач: {len(tasks)}[/green]")
    console.print(f"[green]✓ Мастер-нейронов: {len(masters)}[/green]")
    
    # Загружаем модель один раз
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold]Загрузка модели на {device}...[/bold]")
    
    start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    console.print(f"[green]✓ Модель загружена за {time.time() - start_time:.1f} сек[/green]")
    
    # Сохраняем оригинальные веса
    console.print("[dim]Сохранение оригинальных весов...[/dim]")
    original_weights = copy.deepcopy(model.state_dict())
    
    # Базовый тест
    console.print("\n[bold yellow]━━━ БАЗОВЫЙ ТЕСТ (без усиления) ━━━[/bold yellow]")
    baseline_accuracy = test_model_with_progress(model, tokenizer, tasks, device, "Базовый тест")
    console.print(f"\n[bold]Базовая точность: {baseline_accuracy*100:.1f}%[/bold]\n")
    
    # Коэффициенты для тестирования
    amplifications = [1.1, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5]
    
    results = {'master': [], 'random': []}
    
    for amp in amplifications:
        console.print(f"\n[bold magenta]━━━ КОЭФФИЦИЕНТ {amp} ━━━[/bold magenta]")
        
        # === МАСТЕР-НЕЙРОНЫ ===
        console.print("[cyan]1/2 Тестирование мастер-нейронов...[/cyan]")
        model.load_state_dict(original_weights)
        modified = apply_master_seed(model, masters, amp)
        console.print(f"[dim]  Применено изменений: {modified}[/dim]")
        
        acc = test_model_with_progress(model, tokenizer, tasks, device, f"Мастеры ×{amp}")
        results['master'].append({'amplification': amp, 'accuracy': acc})
        
        delta = acc - baseline_accuracy
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "yellow"
        console.print(f"\n[{delta_color}]Мастеры ×{amp}: {acc*100:.1f}% ({delta*100:+.1f} от базового)[/{delta_color}]")
        
        # === СЛУЧАЙНЫЕ НЕЙРОНЫ ===
        console.print("\n[cyan]2/2 Тестирование случайных нейронов...[/cyan]")
        model.load_state_dict(original_weights)
        modified = apply_random_seed(model, num_neurons=len(masters), amplification=amp)
        console.print(f"[dim]  Применено изменений: {modified}[/dim]")
        
        acc = test_model_with_progress(model, tokenizer, tasks, device, f"Случайные ×{amp}")
        results['random'].append({'amplification': amp, 'accuracy': acc})
        
        delta = acc - baseline_accuracy
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "yellow"
        console.print(f"\n[{delta_color}]Случайные ×{amp}: {acc*100:.1f}% ({delta*100:+.1f} от базового)[/{delta_color}]")
    
    # === ИТОГОВАЯ ТАБЛИЦА ===
    console.print("\n[bold green]═══════════════════════════════════════════════════════[/bold green]")
    console.print("[bold]РЕЗУЛЬТАТЫ[/bold]")
    
    table = Table(title="Сравнение коэффициентов усиления")
    table.add_column("Коэф.", style="cyan", justify="center")
    table.add_column("Мастер-нейроны", style="green", justify="center")
    table.add_column("Δ", style="yellow", justify="center")
    table.add_column("Случайные", style="red", justify="center")
    table.add_column("Δ", style="yellow", justify="center")
    
    for i, amp in enumerate(amplifications):
        master_acc = results['master'][i]['accuracy'] * 100
        master_delta = results['master'][i]['accuracy'] - baseline_accuracy
        random_acc = results['random'][i]['accuracy'] * 100
        random_delta = results['random'][i]['accuracy'] - baseline_accuracy
        
        table.add_row(
            f"{amp}",
            f"{master_acc:.1f}%",
            f"{master_delta*100:+.1f}%",
            f"{random_acc:.1f}%",
            f"{random_delta*100:+.1f}%"
        )
    
    console.print(table)
    
    # График
    plt.figure(figsize=(12, 6))
    
    master_acc = [r['accuracy'] * 100 for r in results['master']]
    random_acc = [r['accuracy'] * 100 for r in results['random']]
    
    plt.plot(amplifications, master_acc, 'o-', label='Мастер-нейроны', linewidth=2, markersize=8, color='green')
    plt.plot(amplifications, random_acc, 's--', label='Случайные нейроны', linewidth=2, markersize=8, color='red')
    plt.axhline(y=baseline_accuracy*100, color='gray', linestyle=':', label=f'Базовая точность ({baseline_accuracy*100:.1f}%)')
    
    plt.xlabel('Коэффициент усиления', fontsize=12)
    plt.ylabel('Точность (%)', fontsize=12)
    plt.title('Влияние усиления на точность арифметических задач', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Добавляем значения на график
    for i, (x, y) in enumerate(zip(amplifications, master_acc)):
        plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    for i, (x, y) in enumerate(zip(amplifications, random_acc)):
        plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
    
    plot_path = base_dir / "results" / f"amplification_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    # Сохраняем JSON
    results_path = base_dir / "results" / f"amplification_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_accuracy,
            'amplifications': amplifications,
            'master_results': results['master'],
            'random_results': results['random'],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    console.print(f"\n[green]✅ График сохранён: {plot_path}[/green]")
    console.print(f"[green]✅ Данные сохранены: {results_path}[/green]")
    
    # Лучший коэффициент
    best_amp = None
    best_acc = baseline_accuracy
    for i, amp in enumerate(amplifications):
        if results['master'][i]['accuracy'] > best_acc:
            best_acc = results['master'][i]['accuracy']
            best_amp = amp
    
    if best_amp:
        console.print(f"\n[bold green]🏆 Лучший результат: ×{best_amp} с точностью {best_acc*100:.1f}%[/bold green]")
    
    console.print(f"\n[dim]Общее время выполнения: {time.time() - start_time:.1f} сек[/dim]")


if __name__ == "__main__":
    main()
