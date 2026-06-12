#!/usr/bin/env python3
"""
Загрузка Alpaca и разделение на задачи Code Generation и Summarization.
"""
import json
import os
from datasets import load_dataset

def filter_code_examples(alpaca_data, max_samples=500):
    """Фильтрует примеры для генерации кода."""
    code_keywords = [
        'code', 'function', 'program', 'python', 'write a',
        'implement', 'algorithm', 'script', 'class', 'def ',
        'create a function', 'write code', 'programming'
    ]
    filtered = []
    for item in alpaca_data:
        instruction = item.get('instruction', '').lower()
        if any(kw in instruction for kw in code_keywords):
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            filtered.append({'text': text})
            if len(filtered) >= max_samples:
                break
    return filtered

def filter_summary_examples(alpaca_data, max_samples=500):
    """Фильтрует примеры для суммаризации."""
    summary_keywords = [
        'summarize', 'summary', 'brief', 'shorten', 'tldr',
        'in short', 'concise', 'summarize the following',
        'give me a brief', 'abstract'
    ]
    filtered = []
    for item in alpaca_data:
        instruction = item.get('instruction', '').lower()
        input_text = item.get('input', '').lower()
        combined = instruction + ' ' + input_text
        if any(kw in combined for kw in summary_keywords):
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            filtered.append({'text': text})
            if len(filtered) >= max_samples:
                break
    return filtered

def main():
    print("=" * 70)
    print(" Подготовка датасета Alpaca для MeaningSeed")
    print("=" * 70)
    
    print("\n[1] Загрузка Alpaca...")
    alpaca = load_dataset("tatsu-lab/alpaca")
    train_data = alpaca['train']
    print(f" Всего примеров: {len(train_data)}")
    
    os.makedirs('data', exist_ok=True)
    
    # Code Generation
    print("\n[2] Фильтрация Code Generation...")
    code_examples = filter_code_examples(train_data, max_samples=500)
    with open('data/alpaca_code.jsonl', 'w', encoding='utf-8') as f:
        for ex in code_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f" Сохранено: {len(code_examples)} примеров → data/alpaca_code.jsonl")
    
    # Summarization
    print("\n[3] Фильтрация Summarization...")
    summary_examples = filter_summary_examples(train_data, max_samples=500)
    with open('data/alpaca_summary.jsonl', 'w', encoding='utf-8') as f:
        for ex in summary_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f" Сохранено: {len(summary_examples)} примеров → data/alpaca_summary.jsonl")
    
    print("\n✅ Готово! Теперь можно запускать extract:")
    print(" python cli.py extract --dataset data/alpaca_code.jsonl --task code --output seeds/code.pt")
    print(" python cli.py extract --dataset data/alpaca_summary.jsonl --task summary --output seeds/summary.pt")

if __name__ == '__main__':
    main()
