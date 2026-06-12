#!/usr/bin/env python3
"""
Демонстрация переключения задач на реальных данных Alpaca.
Code Generation ↔ Summarization
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from meaning_seed import Orchestrator, MeaningExtractor, SeedRegistry

def make_alpaca_loader(filepath, tokenizer, max_samples=200):
    """Загружает Alpaca-подобный JSONL в DataLoader."""
    import json
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            if len(texts) >= max_samples:
                break
    
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    
    class DS(torch.utils.data.Dataset):
        def __init__(self, e): self.e = e
        def __len__(self): return len(self.e['input_ids'])
        def __getitem__(self, i):
            return {
                'input_ids': self.e['input_ids'][i],
                'attention_mask': self.e['attention_mask'][i],
                'labels': self.e['input_ids'][i].clone()
            }
    
    return torch.utils.data.DataLoader(DS(enc), batch_size=8, shuffle=True)

def train_expert(model, dataloader, epochs=2):
    """Fine-tuning только MLP-слоев."""
    for param in model.parameters():
        param.requires_grad = False
    for i in range(len(model.transformer.h)):
        model.transformer.h[i].mlp.c_fc.weight.requires_grad = True
        model.transformer.h[i].mlp.c_proj.weight.requires_grad = True
        model.transformer.h[i].mlp.c_fc.bias.requires_grad = True
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    
    from tqdm import tqdm
    for _ in tqdm(range(epochs), desc=" Обучение эксперта"):
        for batch in dataloader:
            opt.zero_grad()
            out = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
            out.loss.backward()
            opt.step()
    return model

def main():
    print("=" * 70)
    print(" MEANINGSEED: Real-World Proof на Alpaca")
    print(" Code Generation ↔ Summarization")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка данных
    print("\n[1] Загрузка данных Alpaca...")
    loader_code = make_alpaca_loader('data/alpaca_code.jsonl', tokenizer)
    loader_summary = make_alpaca_loader('data/alpaca_summary.jsonl', tokenizer)
    print(f" Code batches: {len(loader_code)}")
    print(f" Summary batches: {len(loader_summary)}")
    
    # Извлечение семян
    print("\n[2] Обучение экспертов и извлечение семян...")
    
    # Code expert
    print(" → Эксперт Code Generation")
    torch.manual_seed(42)
    model_code = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model_code = train_expert(model_code, loader_code, epochs=2)
    extractor = MeaningExtractor()
    masters_code = extractor.extract_distributed_masters(
        model_code, loader_code, masters_per_layer=10
    )
    SeedRegistry.save_seed('seeds/alpaca_code.pt', 'distilgpt2', masters_code, model_code, 'code')
    del model_code
    
    # Summary expert
    print(" → Эксперт Summarization")
    torch.manual_seed(42)
    model_summary = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model_summary = train_expert(model_summary, loader_summary, epochs=2)
    masters_summary = extractor.extract_distributed_masters(
        model_summary, loader_summary, masters_per_layer=10
    )
    SeedRegistry.save_seed('seeds/alpaca_summary.pt', 'distilgpt2', masters_summary, model_summary, 'summary')
    del model_summary
    
    # Оркестрация
    print("\n[3] Демонстрация переключения...")
    orch = Orchestrator('distilgpt2')
    seed_code = SeedRegistry.load_seed('seeds/alpaca_code.pt')
    seed_summary = SeedRegistry.load_seed('seeds/alpaca_summary.pt')
    
    # Тестовые промпты
    test_code = "### Instruction:\nWrite a Python function to calculate fibonacci numbers\n\n### Response:\n"
    test_summary = "### Instruction:\nSummarize the following text\n\n### Input:\nArtificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.\n\n### Response:\n"
    
    # Состояние 0: Чистая нода
    print("\n[Состояние 0: Чистая нода (distilgpt2 base)]")
    print(f" Code ppl:     {orch.evaluate_perplexity(loader_code):.2f}")
    print(f" Summary ppl:  {orch.evaluate_perplexity(loader_summary):.2f}")
    
    # Inject Code
    print("\n[Состояние 1: INJECT Code + Warmup]")
    orch.inject_seed(seed_code, warmup_epochs=1, dataloader=loader_code)
    print(f" Code ppl:     {orch.evaluate_perplexity(loader_code):.2f} 🚀")
    print(f" Summary ppl:  {orch.evaluate_perplexity(loader_summary):.2f}")
    print(f"\n Генерация (Code):")
    print(f" {orch.generate(test_code, max_length=150)}")
    
    # Eject Code
    print("\n[Состояние 2: EJECT Code]")
    orch.eject_seed()
    print(f" Code ppl:     {orch.evaluate_perplexity(loader_code):.2f} 💥")
    
    # Inject Summary
    print("\n[Состояние 3: INJECT Summary + Warmup]")
    orch.inject_seed(seed_summary, warmup_epochs=1, dataloader=loader_summary)
    print(f" Code ppl:     {orch.evaluate_perplexity(loader_code):.2f}")
    print(f" Summary ppl:  {orch.evaluate_perplexity(loader_summary):.2f} 🚀")
    print(f"\n Генерация (Summary):")
    print(f" {orch.generate(test_summary, max_length=150)}")
    
    # Eject Summary
    print("\n[Состояние 4: EJECT Summary]")
    orch.eject_seed()
    print(f" Summary ppl:  {orch.evaluate_perplexity(loader_summary):.2f} 💥")
    
    print("\n" + "=" * 70)
    print(" ✅ Real-World Proof завершён!")
    print("=" * 70)

if __name__ == '__main__':
    main()
