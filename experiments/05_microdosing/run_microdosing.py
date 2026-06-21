#!/usr/bin/env python3
"""
Эксперимент 05: Микродозинг (стохастический резонанс)
С АГРЕССИВНОЙ ОЧИСТКОЙ ПАМЯТИ ДЛЯ CPU
"""

import json
import torch
import gc
import re
import copy
import psutil
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Отключаем прогресс-бары Rich (они жрут память)
# Используем простой print


def log_memory(stage=""):
    """Логирование памяти для отладки"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEM {stage}] {mem_mb:.0f} MB")
    return mem_mb


def load_tasks(tasks_path, limit=50):
    with open(tasks_path) as f:
        tasks = json.load(f)
    return tasks[:limit]


def test_model(model, tokenizer, tasks, device, dose_pct, verbose=False):
    """Тестирует модель с минимальным потреблением памяти"""
    correct = 0
    results = []
    
    for i, task in enumerate(tasks):
        prompt = task['prompt']
        correct_answer = task['answer']
        
        if verbose:
            print(f"  [{i+1}/{len(tasks)}] {prompt[:50]}...", end=" ", flush=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Ищем число
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
        
        if verbose:
            print(f"{'✓' if is_correct else '✗'} (получено {predicted})")
        
        results.append({
            'correct': is_correct,
            'predicted': predicted
        })
        
        # КРИТИЧНО: очищаем всё после каждой задачи
        del inputs
        del outputs
        gc.collect()
        
        # Не вызываем cuda.empty_cache() на CPU
    
    accuracy = correct / len(tasks) if tasks else 0
    return accuracy, results


def add_noise_to_model(model, noise_std):
    """Добавляет гауссов шум ко всем параметрам"""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)


def restore_weights(model, original_weights):
    """Восстанавливает веса из копии"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_weights:
                param.copy_(original_weights[name])


def main():
    print("=" * 60)
    print("Эксперимент 05: Микродозинг (стохастический резонанс)")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    tasks_path = base_dir.parent / "02_arithmetic_masters" / "tasks" / "arithmetic_5digit_fixed.json"
    
    if not tasks_path.exists():
        print("Ошибка: задачи не найдены")
        return
    
    tasks = load_tasks(tasks_path, 30)  # УМЕНЬШИЛ до 30 задач (было 50)
    print(f"Загружено {len(tasks)} задач")
    
    device = "cpu"  # Принудительно CPU
    print(f"Устройство: {device}")
    
    # Загружаем модель
    print("\nЗагрузка модели...")
    log_memory("до загрузки")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,  # CPU не любит float16
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_memory("после загрузки")
    
    # Сохраняем веса (только нужные, не весь state_dict)
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.clone()
    
    log_memory("после копирования весов")
    
    # График доз (уменьшил количество)
    dose_schedule = [0.0, 0.005, 0.01, 0.02, 0.03]
    all_results = []
    
    # === БАЗОВЫЙ ТЕСТ ===
    print("\n--- БАЗОВЫЙ ТЕСТ (без шума) ---")
    baseline_accuracy, baseline_results = test_model(model, tokenizer, tasks, device, 0, verbose=True)
    print(f"Базовая точность: {baseline_accuracy*100:.1f}%")
    
    all_results.append({
        'dose': 0.0,
        'accuracy': baseline_accuracy,
        'results': baseline_results
    })
    
    gc.collect()
    log_memory("после базового теста")
    
    # === ТЕСТЫ С ДОЗАМИ ===
    for dose in dose_schedule[1:]:
        print(f"\n--- ДОЗА: {dose*100:.2f}% шума ---")
        log_memory(f"перед дозой {dose}")
        
        # Восстанавливаем веса
        restore_weights(model, original_weights)
        gc.collect()
        
        # Добавляем шум
        add_noise_to_model(model, dose)
        
        # Тестируем
        accuracy, results = test_model(model, tokenizer, tasks, device, dose, verbose=True)
        print(f"Точность: {accuracy*100:.1f}%")
        
        delta = accuracy - baseline_accuracy
        if delta > 0:
            print(f"✅ УЛУЧШЕНИЕ: +{delta*100:.1f}%")
        elif delta < 0:
            print(f"❌ УХУДШЕНИЕ: {delta*100:.1f}%")
        else:
            print(f"⚠️ БЕЗ ИЗМЕНЕНИЙ")
        
        all_results.append({
            'dose': dose,
            'accuracy': accuracy,
            'delta': delta,
            'results': results
        })
        
        # Агрессивная очистка
        del results
        gc.collect()
        log_memory(f"после дозы {dose}")
        
        # Стоп-условие
        if accuracy < baseline_accuracy * 0.3 and baseline_accuracy > 0:
            print(f"\n⚠️ КОГНИТИВНЫЙ КОЛЛАПС, останавливаюсь")
            break
    
    # === ИТОГИ ===
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 60)
    print(f"{'Доза (%)':<12} {'Точность (%)':<15} {'Изменение (%)':<15}")
    print("-" * 42)
    
    best_dose = 0.0
    best_accuracy = baseline_accuracy
    
    for r in all_results:
        dose_pct = r['dose'] * 100
        acc_pct = r['accuracy'] * 100
        delta_pct = r.get('delta', 0) * 100
        
        if r['accuracy'] > best_accuracy:
            best_accuracy = r['accuracy']
            best_dose = r['dose']
        
        print(f"{dose_pct:<12.2f} {acc_pct:<15.1f} {delta_pct:+.1f}")
    
    print("-" * 42)
    
    if best_dose > 0:
        print(f"\n✨ ОПТИМАЛЬНАЯ ДОЗА: {best_dose*100:.2f}%")
        print("🎯 ВЕРДИКТ: СТОХАСТИЧЕСКИЙ РЕЗОНАНС")
    else:
        print(f"\n💀 ОПТИМАЛЬНАЯ ДОЗА: 0%")
        print("💀 ВЕРДИКТ: МИКРОДОЗИНГ БЕСПОЛЕЗЕН")
    
    # Сохраняем результат
    result_path = base_dir / "reports" / "microdosing_result.json"
    result_path.parent.mkdir(exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_accuracy,
            'best_accuracy': best_accuracy,
            'best_dose': best_dose,
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\nРезультат сохранён: {result_path}")


if __name__ == "__main__":
    main()
