#!/usr/bin/env python3
"""
Демонстрация мгновенного переключения задач через MeaningSeed.
Повторяет эксперимент из BrainTopologyLLM, но через чистый API.
"""

import random
import torch
from transformers import GPT2Tokenizer

from meaning_seed import Orchestrator, MeaningExtractor, SeedRegistry


def make_json_dataset(tokenizer, n=400):
    texts = [f'{{"id": {i}, "status": "ok", "code": 200}}' for i in range(n)]
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    class DS(torch.utils.data.Dataset):
        def __init__(self, e): self.e = e
        def __len__(self): return len(self.e['input_ids'])
        def __getitem__(self, i):
            return {'input_ids': self.e['input_ids'][i],
                    'attention_mask': self.e['attention_mask'][i],
                    'labels': self.e['input_ids'][i].clone()}
    return DS(enc)


def make_math_dataset(tokenizer, n=400):
    texts = []
    for _ in range(n):
        a, b = random.randint(1, 50), random.randint(1, 50)
        texts.append(f'Q: {a} + {b} = ?\nA: {a+b}')
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    class DS(torch.utils.data.Dataset):
        def __init__(self, e): self.e = e
        def __len__(self): return len(self.e['input_ids'])
        def __getitem__(self, i):
            return {'input_ids': self.e['input_ids'][i],
                    'attention_mask': self.e['attention_mask'][i],
                    'labels': self.e['input_ids'][i].clone()}
    return DS(enc)


def train_expert(model, dataloader, epochs=2):
    for param in model.parameters():
        param.requires_grad = False
    for i in range(len(model.transformer.h)):
        model.transformer.h[i].mlp.c_fc.weight.requires_grad = True
        model.transformer.h[i].mlp.c_proj.weight.requires_grad = True
        model.transformer.h[i].mlp.c_fc.bias.requires_grad = True
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    from tqdm import tqdm
    for _ in tqdm(range(epochs), desc="  Обучение эксперта"):
        for batch in dataloader:
            opt.zero_grad()
            out = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
            out.loss.backward()
            opt.step()
    return model


def main():
    print("=" * 70)
    print(" MEANINGSEED: Демонстрация переключения задач")
    print("=" * 70)
    
    # 1. Подготовка экспертов
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    ds_json = make_json_dataset(tokenizer)
    ds_math = make_math_dataset(tokenizer)
    loader_json = torch.utils.data.DataLoader(ds_json, batch_size=16, shuffle=True)
    loader_math = torch.utils.data.DataLoader(ds_math, batch_size=16, shuffle=True)
    
    print("\n[1] Обучение экспертов и извлечение семян...")
    
    from transformers import GPT2LMHeadModel
    
    # Эксперт JSON
    print("  -> Эксперт JSON")
    torch.manual_seed(42)
    model_json = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model_json = train_expert(model_json, loader_json)
    extractor = MeaningExtractor()
    masters_json = extractor.extract_distributed_masters(model_json, loader_json, masters_per_layer=10)
    SeedRegistry.save_seed('seeds/json_expert.pt', 'distilgpt2', masters_json, model_json, 'json')
    del model_json
    
    # Эксперт Math
    print("  -> Эксперт Math")
    torch.manual_seed(42)
    model_math = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model_math = train_expert(model_math, loader_math)
    masters_math = extractor.extract_distributed_masters(model_math, loader_math, masters_per_layer=10)
    SeedRegistry.save_seed('seeds/math_expert.pt', 'distilgpt2', masters_math, model_math, 'math')
    del model_math
    
    # 2. Оркестрация
    print("\n[2] Демонстрация переключения...")
    orch = Orchestrator('distilgpt2')
    
    seed_json = SeedRegistry.load_seed('seeds/json_expert.pt')
    seed_math = SeedRegistry.load_seed('seeds/math_expert.pt')
    
    # Чистая нода
    print("\n[Состояние 0: Чистая нода]")
    print(f"  JSON ppl: {orch.evaluate_perplexity(loader_json):.2f}")
    print(f"  Math ppl: {orch.evaluate_perplexity(loader_math):.2f}")
    
    # Inject JSON
    print("\n[Состояние 1: INJECT JSON + Warmup]")
    orch.inject_seed(seed_json, warmup_epochs=1, dataloader=loader_json)
    print(f"  JSON ppl: {orch.evaluate_perplexity(loader_json):.2f} 🚀")
    print(f"  Math ppl: {orch.evaluate_perplexity(loader_math):.2f}")
    
    # Eject
    print("\n[Состояние 2: EJECT JSON]")
    orch.eject_seed()
    print(f"  JSON ppl: {orch.evaluate_perplexity(loader_json):.2f} 💥")
    
    # Inject Math
    print("\n[Состояние 3: INJECT Math + Warmup]")
    orch.inject_seed(seed_math, warmup_epochs=1, dataloader=loader_math)
    print(f"  JSON ppl: {orch.evaluate_perplexity(loader_json):.2f}")
    print(f"  Math ppl: {orch.evaluate_perplexity(loader_math):.2f} 🚀")
    
    # Eject
    print("\n[Состояние 4: EJECT Math]")
    orch.eject_seed()
    print(f"  Math ppl: {orch.evaluate_perplexity(loader_math):.2f} 💥")
    
    print("\n" + "=" * 70)
    print(" ✅ Демонстрация завершена!")
    print("=" * 70)


if __name__ == '__main__':
    main()
