#!/usr/bin/env python3
"""
Проращивание семени в Qwen Base
Усиливает мастер-нейроны из Instruct в Base модели
"""

import json
import torch
import gc
import re
import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_seed(seed_path: Path):
    with open(seed_path) as f:
        return json.load(f)


def load_tasks(tasks_path: Path):
    with open(tasks_path) as f:
        return json.load(f)


def test_model(model, tokenizer, tasks, device, max_tokens=50):
    """Тестирует модель на задачах и возвращает точность"""
    correct = 0
    
    for i, task in enumerate(tasks):
        prompt = task['prompt']
        correct_answer = task['answer']
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Ищем число в ответе
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            try:
                predicted = int(numbers[-1])
                if predicted == correct_answer:
                    correct += 1
            except:
                pass
        
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if (i + 1) % 20 == 0:
            console.print(f"  Прогресс: {i+1}/{len(tasks)} (правильных: {correct})")
    
    return correct / len(tasks) if tasks else 0


def apply_seed(model, seed, amplification=1.3):
    """Усиливает мастер-нейроны в модели"""
    masters = seed.get('masters', [])
    modified = 0
    layers_modified = set()
    
    for m in masters:
        layer_name = m['layer']
        neuron_idx = m['neuron']
        
        # Находим модуль
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
                layers_modified.add(layer_name)
                if hasattr(module, 'bias') and module.bias is not None:
                    if neuron_idx < module.bias.shape[0]:
                        module.bias.data[neuron_idx] *= amplification
    
    return modified, len(layers_modified)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplify", type=float, default=1.3, help="Коэффициент усиления")
    parser.add_argument("--tasks", type=int, default=100, help="Количество задач для теста (max 100)")
    args = parser.parse_args()
    
    console.print(Panel.fit(f"Проращивание семени в Qwen Base (amplification = {args.amplify})", border_style="cyan"))
    
    # Пути
    base_dir = Path(__file__).parent
    seed_path = base_dir / "masters" / "arithmetic_seed.json"
    tasks_path = base_dir / "tasks" / "arithmetic_5digit_fixed.json"
    
    # Загрузка
    seed = load_seed(seed_path)
    all_tasks = load_tasks(tasks_path)
    tasks = all_tasks[:args.tasks]
    
    console.print(f"[dim]Семя: {len(seed['masters'])} мастеров из {seed['model_source']}[/dim]")
    console.print(f"[dim]Задач: {len(tasks)}[/dim]")
    console.print(f"[dim]Коэффициент усиления: {args.amplify}[/dim]")
    
    # Загрузка Base модели
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold]Загрузка Qwen Base на {device}...[/bold]")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Тест ДО
    console.print("\n[bold]Тест ДО проращивания...[/bold]")
    accuracy_before = test_model(model, tokenizer, tasks, device)
    console.print(f"[yellow]Точность Base ДО: {accuracy_before*100:.1f}%[/yellow]")
    
    # Применяем семя
    console.print(f"\n[bold]🌱 Проращивание семени (×{args.amplify})...[/bold]")
    modified, layers_modified = apply_seed(model, seed, amplification=args.amplify)
    
    if modified == 0:
        console.print("[red]Не удалось применить семя. Проверь названия слоёв.[/red]")
        return
    
    console.print(f"[green]Усилено мастер-нейронов: {modified}/{len(seed['masters'])}[/green]")
    console.print(f"[dim]Затронуто слоёв: {layers_modified}[/dim]")
    
    # Тест ПОСЛЕ
    console.print("\n[bold]Тест ПОСЛЕ проращивания...[/bold]")
    accuracy_after = test_model(model, tokenizer, tasks, device)
    console.print(f"[green]Точность Base ПОСЛЕ: {accuracy_after*100:.1f}%[/green]")
    
    # Результат
    delta = accuracy_after - accuracy_before
    delta_percent = (accuracy_after / accuracy_before - 1) * 100 if accuracy_before > 0 else float('inf')
    
    if delta > 0:
        console.print(f"\n[bold green]✅ УЛУЧШЕНИЕ: +{delta*100:.1f}% (относительный прирост: {delta_percent:.1f}%)[/bold green]")
    elif delta < 0:
        console.print(f"\n[bold red]❌ УХУДШЕНИЕ: {delta*100:.1f}%[/bold red]")
    else:
        console.print(f"\n[bold yellow]⚠️ БЕЗ ИЗМЕНЕНИЙ[/bold yellow]")
    
    # Сохраняем результат
    result = {
        'seed': str(seed_path),
        'amplification': args.amplify,
        'tasks_tested': len(tasks),
        'accuracy_before': accuracy_before,
        'accuracy_after': accuracy_after,
        'delta': delta,
        'relative_improvement': delta_percent if accuracy_before > 0 else None,
        'masters_applied': modified,
        'layers_modified': layers_modified
    }
    
    result_path = base_dir / "results" / f"germination_amp{args.amplify}.json"
    result_path.parent.mkdir(exist_ok=True)
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    console.print(f"\n[dim]Результат сохранён: {result_path}[/dim]")


if __name__ == "__main__":
    main()
