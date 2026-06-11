#!/usr/bin/env python3
"""
MeaningSeed CLI: Точка входа для оркестрации задач.

Использование:
    python cli.py extract --model distilgpt2 --dataset data/math.json --output seeds/math.pt
    python cli.py inject --model distilgpt2 --seed seeds/math.pt --warmup 1
    python cli.py list
"""

import argparse
import json
import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from meaning_seed import Orchestrator, MeaningExtractor, SeedRegistry


def load_dataset_from_json(filepath: str, tokenizer: GPT2Tokenizer, num_samples: int = 400):
    """Загружает датасет из JSON/JSONL файла."""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Поддерживаем разные форматы
                text = data.get('text') or data.get('instruction', '') + data.get('output', '')
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
            
            if len(texts) >= num_samples:
                break
    
    if not texts:
        raise ValueError(f"Не удалось загрузить тексты из {filepath}")
    
    print(f"  Загружено {len(texts)} примеров из {filepath}")
    
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return len(self.enc['input_ids'])
        def __getitem__(self, idx):
            return {
                'input_ids': self.enc['input_ids'][idx],
                'attention_mask': self.enc['attention_mask'][idx],
                'labels': self.enc['input_ids'][idx].clone()
            }
    
    return SimpleDataset(enc)


def cmd_extract(args):
    """Команда: извлечение семени из обученной модели."""
    print("=" * 70)
    print(f" MEANINGSEED: Извлечение семени для задачи '{args.task}'")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка датасета
    print("\n[1] Загрузка данных...")
    dataset = load_dataset_from_json(args.dataset, tokenizer, num_samples=args.samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Обучение модели (если нужно)
    print("\n[2] Подготовка модели...")
    if args.pretrained_model:
        print(f"  Загрузка обученной модели: {args.pretrained_model}")
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:
        print(f"  Обучение базовой модели {args.model} на задаче...")
        model = GPT2LMHeadModel.from_pretrained(args.model)
        # Простой fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        for i in range(len(model.transformer.h)):
            model.transformer.h[i].mlp.c_fc.weight.requires_grad = True
            model.transformer.h[i].mlp.c_proj.weight.requires_grad = True
            model.transformer.h[i].mlp.c_fc.bias.requires_grad = True
        
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
        from tqdm import tqdm
        for epoch in tqdm(range(args.epochs), desc="  Обучение"):
            for batch in dataloader:
                opt.zero_grad()
                out = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
                out.loss.backward()
                opt.step()
    
    # Извлечение мастеров
    print("\n[3] Извлечение топологических мастеров...")
    extractor = MeaningExtractor()
    masters = extractor.extract_distributed_masters(
        model, dataloader, masters_per_layer=args.masters_per_layer
    )
    
    # Сохранение
    print("\n[4] Сохранение семени...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    SeedRegistry.save_seed(
        filepath=args.output,
        model_name=args.model,
        masters_per_layer=masters,
        model=model,
        task_name=args.task
    )
    
    print("\n✅ Готово!")


def cmd_inject(args):
    """Команда: внедрение семени в модель."""
    print("=" * 70)
    print(" MEANINGSEED: Внедрение семени")
    print("=" * 70)
    
    seed_data = SeedRegistry.load_seed(args.seed)
    orch = Orchestrator(model_name=args.model)
    
    dataloader = None
    if args.dataset:
        print("\n[1] Загрузка данных для warmup...")
        tokenizer = orch.tokenizer
        dataset = load_dataset_from_json(args.dataset, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print("\n[2] Внедрение семени...")
    orch.inject_seed(seed_data, warmup_epochs=args.warmup, dataloader=dataloader)
    
    if args.test_prompt:
        print(f"\n[3] Тестовая генерация:")
        print(f"  Prompt: {args.test_prompt}")
        result = orch.generate(args.test_prompt, max_length=args.max_length)
        print(f"  Output: {result}")
    
    if args.output_model:
        print(f"\n[4] Сохранение модели с внедренным семенем: {args.output_model}")
        orch.model.save_pretrained(args.output_model)
    
    print("\n✅ Готово!")


def cmd_list(args):
    """Команда: список доступных семян."""
    print("=" * 70)
    print(" MEANINGSEED: Доступные семена")
    print("=" * 70)
    
    seeds = SeedRegistry.list_seeds(args.directory)
    if not seeds:
        print("  Семян не найдено.")
        return
    
    for seed_file in seeds:
        filepath = os.path.join(args.directory, seed_file)
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"\n  📦 {seed_file}")
        print(f"     Задача: {data.get('task_name', 'unknown')}")
        print(f"     Модель: {data.get('model_name', 'unknown')}")
        print(f"     Размер: {data.get('seed_size_kb', 0):.2f} KB")
        print(f"     Мастеров: {data.get('total_masters', 0)}")
        print(f"     Создано: {data.get('created_at', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(
        prog='meaningseed',
        description='MeaningSeed: Топологическая оркестрация задач LLM'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Команда extract
    p_extract = subparsers.add_parser('extract', help='Извлечь семя из модели')
    p_extract.add_argument('--model', default='distilgpt2', help='Базовая модель')
    p_extract.add_argument('--dataset', required=True, help='Путь к JSONL файлу с задачей')
    p_extract.add_argument('--task', required=True, help='Имя задачи (для метаданных)')
    p_extract.add_argument('--output', required=True, help='Путь для сохранения семени')
    p_extract.add_argument('--masters-per-layer', type=int, default=10)
    p_extract.add_argument('--samples', type=int, default=400)
    p_extract.add_argument('--batch-size', type=int, default=16)
    p_extract.add_argument('--epochs', type=int, default=2, help='Эпохи обучения (если нет pretrained)')
    p_extract.add_argument('--pretrained-model', help='Путь к уже обученной модели (опционально)')
    
    # Команда inject
    p_inject = subparsers.add_parser('inject', help='Внедрить семя в модель')
    p_inject.add_argument('--model', default='distilgpt2', help='Базовая модель')
    p_inject.add_argument('--seed', required=True, help='Путь к файлу семени')
    p_inject.add_argument('--warmup', type=int, default=1, help='Эпохи целевого прогрева')
    p_inject.add_argument('--dataset', help='Датасет для warmup (опционально)')
    p_inject.add_argument('--batch-size', type=int, default=16)
    p_inject.add_argument('--test-prompt', help='Тестовый промпт для проверки')
    p_inject.add_argument('--max-length', type=int, default=50)
    p_inject.add_argument('--output-model', help='Сохранить модель с семенем (опционально)')
    
    # Команда list
    p_list = subparsers.add_parser('list', help='Список доступных семян')
    p_list.add_argument('--directory', default='seeds', help='Директория с семенами')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        cmd_extract(args)
    elif args.command == 'inject':
        cmd_inject(args)
    elif args.command == 'list':
        cmd_list(args)


if __name__ == '__main__':
    main()
