#!/usr/bin/env python3
"""
Томограф для арифметики v2 - правильный формат для Qwen Instruct
Использует задачи из tasks/arithmetic_5digit_fixed.json
"""

import json
import torch
import gc
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

console = Console()


def is_correct_arithmetic(response: str, correct_answer: int) -> bool:
    """Проверяет, правильный ли ответ в генерации модели"""
    # Ищем числа в ответе
    numbers = re.findall(r'-?\d+', response)
    if not numbers:
        return False
    
    # Берём последнее число (обычно ответ)
    try:
        predicted = int(numbers[-1])
        return predicted == correct_answer
    except:
        return False


def extract_answer_from_generation(full_response: str) -> str:
    """Извлекает только часть после ассистента для отладки"""
    if "<|im_start|>assistant" in full_response:
        parts = full_response.split("<|im_start|>assistant")
        if len(parts) > 1:
            return parts[-1][:100]
    return full_response[:100]


def main():
    # Загружаем задачи в правильном формате
    tasks_path = Path(__file__).parent / "tasks" / "arithmetic_5digit_fixed.json"
    
    if not tasks_path.exists():
        console.print("[red]Ошибка: файл tasks/arithmetic_5digit_fixed.json не найден[/red]")
        console.print("[yellow]Запусти fix_prompts.py сначала[/yellow]")
        return
    
    with open(tasks_path) as f:
        tasks = json.load(f)
    
    console.print(f"[green]Загружено {len(tasks)} задач в Qwen формате[/green]")
    console.print(f"[dim]Пример промпта: {tasks[0]['prompt'][:80]}...[/dim]")
    
    # Загружаем модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Загрузка Qwen Instruct на {device}...[/dim]")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Собираем активации только на правильных ответах
    layer_stats = {}  # {layer_name: {mean: tensor, m2: tensor, count: int}}
    
    # Регистрируем хуки
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            
            acts = output.detach().cpu()
            # Усредняем по батчу и последовательности
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
    
    # Вешаем хуки на все линейные слои
    for name, module in model.named_modules():
        if 'Linear' in str(module.__class__):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Прогоняем задачи
    correct_count = 0
    correct_tasks = []
    console.print("\n[bold]Прогон задач...[/bold]")
    
    for i, task in enumerate(tasks):
        prompt = task['prompt']
        correct_answer = task['answer']
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # низкая температура для детерминизма
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Проверяем ответ
        if is_correct_arithmetic(full_response, correct_answer):
            correct_count += 1
            correct_tasks.append(task)
            
            # Отладочный вывод для первых правильных
            if correct_count <= 3:
                response_preview = extract_answer_from_generation(full_response)
                console.print(f"[green]✓ Задача {i+1}: {task['original_prompt']} -> {correct_answer}[/green]")
                console.print(f"[dim]  Ответ модели: {response_preview}[/dim]")
        else:
            # Отладочный вывод для первых неправильных
            if i < 5:
                response_preview = extract_answer_from_generation(full_response)
                console.print(f"[red]✗ Задача {i+1}: {task['original_prompt']} -> ожидалось {correct_answer}[/red]")
                console.print(f"[dim]  Ответ модели: {response_preview}[/dim]")
        
        # Чистим память
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if (i + 1) % 20 == 0:
            console.print(f"  Прогресс: {i+1}/{len(tasks)} (правильных: {correct_count})")
    
    # Снимаем хуки
    for hook in hooks:
        hook.remove()
    
    console.print(f"\n[green]Правильных ответов: {correct_count}/{len(tasks)}[/green]")
    
    if correct_count < 10:
        console.print("[red]Слишком мало правильных ответов.[/red]")
        console.print("[yellow]Возможные причины:[/yellow]")
        console.print("  1. Модель не понимает формат 'Compute X'")
        console.print("  2. Нужно больше токенов для ответа")
        console.print("  3. Задача слишком сложная для 0.5B модели")
        return
    
    # Вычисляем дисперсию и ищем мастеров
    masters = []
    
    for name, stats in layer_stats.items():
        if stats['count'] < 5:
            continue
            
        variance = stats['m2'] / (stats['count'] - 1) if stats['count'] > 1 else torch.zeros_like(stats['m2'])
        
        mean = stats['mean']
        var = variance
        
        for idx in range(len(mean)):
            mean_val = float(mean[idx])
            var_val = float(var[idx])
            
            # Мастер: высокая средняя (>0.05) и низкая дисперсия (<0.5)
            if mean_val > 0.05 and var_val < 0.5:
                masters.append({
                    'layer': name,
                    'neuron': idx,
                    'mean': mean_val,
                    'variance': var_val
                })
    
    # Сортируем по убыванию средней
    masters.sort(key=lambda x: x['mean'], reverse=True)
    masters = masters[:60]  # берём топ-60
    
    console.print(f"\n[bold green]Найдено мастеров: {len(masters)}[/bold green]")
    
    if masters:
        # Показываем топ-15
        table = Table(title="Топ-15 мастер-нейронов (арифметика)")
        table.add_column("№", style="cyan")
        table.add_column("Слой", style="white")
        table.add_column("Нейрон", style="green")
        table.add_column("Средняя", style="yellow")
        table.add_column("Дисперсия", style="dim")
        
        for i, m in enumerate(masters[:15], 1):
            layer_short = m['layer'].split('.')[-1] if '.' in m['layer'] else m['layer']
            table.add_row(str(i), layer_short, str(m['neuron']), f"{m['mean']:.4f}", f"{m['variance']:.6f}")
        
        console.print(table)
    else:
        console.print("[yellow]Мастеров не найдено. Попробуй уменьшить пороги:[/yellow]")
        console.print("  min_mean = 0.01, max_variance = 1.0")
    
    # Сохраняем семя
    seed = {
        'version': 'arithmetic_02',
        'model_source': 'Qwen/Qwen2.5-0.5B-Instruct',
        'task': 'arithmetic_5digit',
        'num_correct': correct_count,
        'total_tasks': len(tasks),
        'masters': masters,
        'params': {
            'min_mean': 0.05,
            'max_variance': 0.5,
            'temperature': 0.1
        }
    }
    
    seed_path = Path(__file__).parent / "masters" / "arithmetic_seed.json"
    seed_path.parent.mkdir(exist_ok=True)
    
    with open(seed_path, 'w') as f:
        json.dump(seed, f, indent=2)
    
    console.print(f"\n[green]✅ Семя сохранено: {seed_path}[/green]")


if __name__ == "__main__":
    main()
