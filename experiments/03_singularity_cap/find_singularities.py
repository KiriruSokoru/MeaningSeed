#!/usr/bin/env python3
"""
Эксперимент 03: Поиск и ослабление сингулярностей (крикунов)
Детальный вывод по каждой задаче
"""

import json
import torch
import gc
import re
import copy
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()


def load_tasks(tasks_path: Path, limit=50):
    with open(tasks_path) as f:
        tasks = json.load(f)
    return tasks[:limit]


def test_model_with_details(model, tokenizer, tasks, device, description="Тест"):
    """Тестирует модель с детальным выводом по каждой задаче"""
    correct = 0
    results = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_progress = progress.add_task(f"[cyan]{description}", total=len(tasks))
        
        for i, task in enumerate(tasks):
            prompt = task['prompt']
            correct_answer = task['answer']
            
            # Показываем текущую задачу
            progress.update(task_progress, description=f"[cyan]{description} [{i+1}/{len(tasks)}] {prompt[:60]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Ищем число в ответе
            numbers = re.findall(r'-?\d+', response)
            is_correct = False
            predicted = None
            
            if numbers:
                try:
                    predicted = int(numbers[-1])
                    if predicted == correct_answer:
                        is_correct = True
                        correct += 1
                except:
                    pass
            
            # Красивый вывод
            if is_correct:
                console.print(f"  [green]✓[/green] {prompt} = {correct_answer} [dim](получено {predicted})[/dim]")
            else:
                console.print(f"  [red]✗[/red] {prompt} = {correct_answer} [dim](получено {predicted if predicted else 'нет числа'})[/dim]")
            
            results.append({
                'prompt': prompt,
                'correct': is_correct,
                'predicted': predicted,
                'answer': correct_answer
            })
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            progress.update(task_progress, advance=1)
    
    accuracy = correct / len(tasks) if tasks else 0
    return accuracy, results


def collect_activations_with_details(model, tokenizer, tasks, device):
    """Собирает активации с детальным прогрессом"""
    layer_stats = {}
    
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            
            acts = output.detach().cpu()
            while acts.dim() > 1:
                acts = acts.mean(dim=0)
            
            if name not in layer_stats:
                layer_stats[name] = {
                    'mean': torch.zeros_like(acts),
                    'm2': torch.zeros_like(acts),
                    'count': 0
                }
            
            stats = layer_stats[name]
            stats['count'] += 1
            delta = acts - stats['mean']
            stats['mean'] += delta / stats['count']
            delta2 = acts - stats['mean']
            stats['m2'] += delta * delta2
            
            del acts
        return hook
    
    # Регистрируем хуки
    for name, module in model.named_modules():
        if 'Linear' in str(module.__class__):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Прогоняем задачи
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_progress = progress.add_task("[cyan]Сбор активаций", total=len(tasks))
        
        for i, task in enumerate(tasks):
            prompt = task['prompt']
            progress.update(task_progress, description=f"[cyan]Сбор активаций [{i+1}/{len(tasks)}] {prompt[:60]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            progress.update(task_progress, advance=1)
    
    # Снимаем хуки
    for hook in hooks:
        hook.remove()
    
    return layer_stats


def find_singularities_from_stats(layer_stats, variance_threshold=0.5):
    """Из собранной статистики находит крикунов"""
    singularities = []
    
    for name, stats in layer_stats.items():
        if stats['count'] < 5:
            continue
        
        variance = stats['m2'] / (stats['count'] - 1)
        mean = stats['mean']
        
        for idx in range(len(mean)):
            var_val = float(variance[idx])
            mean_val = float(mean[idx])
            
            if var_val > variance_threshold:
                singularities.append({
                    'layer': name,
                    'neuron': idx,
                    'variance': var_val,
                    'mean': mean_val
                })
    
    singularities.sort(key=lambda x: x['variance'], reverse=True)
    return singularities


def apply_cap(model, singularities, cap_factor=0.5):
    """Ослабляет каждого крикуна"""
    modified = 0
    
    for s in singularities:
        layer_name = s['layer']
        neuron_idx = s['neuron']
        
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
                original = module.weight.data[neuron_idx, :].clone()
                module.weight.data[neuron_idx, :] *= cap_factor
                modified += 1
                
                if hasattr(module, 'bias') and module.bias is not None:
                    if neuron_idx < module.bias.shape[0]:
                        module.bias.data[neuron_idx] *= cap_factor
    
    return modified


def main():
    console.print(Panel.fit(
        "[bold cyan]Эксперимент 03: Санация сингулярностей[/bold cyan]\n"
        "Находим крикунов → надеваем колпачок → проверяем точность",
        border_style="cyan"
    ))
    
    base_dir = Path(__file__).parent
    tasks_path = base_dir.parent / "02_arithmetic_masters" / "tasks" / "arithmetic_5digit_fixed.json"
    
    if not tasks_path.exists():
        console.print("[red]Задачи не найдены![/red]")
        return
    
    tasks = load_tasks(tasks_path, 50)
    console.print(f"[green]✓ Загружено {len(tasks)} задач[/green]")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Устройство: {device}[/dim]")
    
    # Загружаем модель
    console.print("\n[bold]Загрузка Qwen Base...[/bold]")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Сохраняем оригинальные веса для отката
    original_weights = copy.deepcopy(model.state_dict())
    
    # === БАЗОВЫЙ ТЕСТ ===
    console.print("\n[bold yellow]━━━ БАЗОВЫЙ ТЕСТ (до колпачка) ━━━[/bold yellow]")
    baseline_accuracy, baseline_results = test_model_with_details(model, tokenizer, tasks, device, "Базовый тест")
    console.print(f"\n[bold]Базовая точность: {baseline_accuracy*100:.1f}%[/bold]")
    
    # === СБОР АКТИВАЦИЙ ===
    console.print("\n[bold]━━━ ПОИСК КРИКУНОВ ━━━[/bold]")
    layer_stats = collect_activations_with_details(model, tokenizer, tasks, device)
    
    # Находим сингулярности
    singularities = find_singularities_from_stats(layer_stats, variance_threshold=0.5)
    console.print(f"\n[red]Найдено крикунов: {len(singularities)}[/red]")
    
    if singularities:
        table = Table(title="Топ-20 крикунов (по дисперсии)")
        table.add_column("№", style="cyan")
        table.add_column("Слой", style="white")
        table.add_column("Нейрон", style="green")
        table.add_column("Дисперсия", style="red")
        table.add_column("Средняя", style="dim")
        
        for i, s in enumerate(singularities[:20], 1):
            layer_short = s['layer'].split('.')[-1] if '.' in s['layer'] else s['layer']
            table.add_row(
                str(i),
                layer_short[:30],
                str(s['neuron']),
                f"{s['variance']:.4f}",
                f"{s['mean']:.4f}"
            )
        console.print(table)
    
    # === НАДЕВАЕМ КОЛПАЧОК ===
    console.print("\n[bold red]━━━ НАДЕВАЕМ КОЛПАЧОК ━━━[/bold red]")
    
    # Берём топ-60 крикунов для ослабления
    top_singularities = singularities[:60]
    console.print(f"[dim]Ослабляем {len(top_singularities)} крикунов с коэффициентом 0.5[/dim]")
    
    modified = apply_cap(model, top_singularities, cap_factor=0.5)
    console.print(f"[green]Ослаблено нейронов: {modified}[/green]")
    
    # === ТЕСТ ПОСЛЕ КОЛПАЧКА ===
    console.print("\n[bold yellow]━━━ ТЕСТ ПОСЛЕ КОЛПАЧКА ━━━[/bold yellow]")
    after_accuracy, after_results = test_model_with_details(model, tokenizer, tasks, device, "Тест после колпачка")
    console.print(f"\n[bold]Точность после: {after_accuracy*100:.1f}%[/bold]")
    
    # === РЕЗУЛЬТАТ ===
    delta = after_accuracy - baseline_accuracy
    if delta > 0:
        console.print(f"\n[bold green]✅ УЛУЧШЕНИЕ: +{delta*100:.1f}%[/bold green]")
    elif delta < 0:
        console.print(f"\n[bold red]❌ УХУДШЕНИЕ: {delta*100:.1f}%[/bold red]")
    else:
        console.print(f"\n[bold yellow]⚠️ БЕЗ ИЗМЕНЕНИЙ[/bold yellow]")
    
    # Детальное сравнение
    console.print("\n[bold]Сравнение ответов (до vs после):[/bold]")
    for i, (before, after) in enumerate(zip(baseline_results[:10], after_results[:10])):
        before_mark = "✓" if before['correct'] else "✗"
        after_mark = "✓" if after['correct'] else "✗"
        console.print(f"  {i+1}. {before_mark} → {after_mark}  |  {before['prompt'][:50]}...")
    
    # Сохраняем результаты
    result = {
        'experiment': '03_singularity_cap',
        'baseline_accuracy': baseline_accuracy,
        'after_accuracy': after_accuracy,
        'delta': delta,
        'singularities_found': len(singularities),
        'singularities_capped': modified,
        'cap_factor': 0.5,
        'variance_threshold': 0.5
    }
    
    result_path = base_dir / "results" / "cap_result.json"
    result_path.parent.mkdir(exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    console.print(f"\n[dim]Результат сохранён: {result_path}[/dim]")
    
    # Восстанавливаем веса (опционально)
    # model.load_state_dict(original_weights)


if __name__ == "__main__":
    main()
