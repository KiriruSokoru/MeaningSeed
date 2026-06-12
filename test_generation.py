#!/usr/bin/env python3
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from meaning_seed import Orchestrator, SeedRegistry

def main():
    print("🔧 Тестирование генерации с загруженными семенами...")
    
    # 1. Инициализация
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    orch = Orchestrator('distilgpt2')
    orch.model.to(device)
    
    # 2. Загрузка семени (например, Code)
    print("\n📥 Загрузка Code Seed...")
    seed_code = SeedRegistry.load_seed('seeds/alpaca_code.pt')
    orch.inject_seed(seed_code, warmup_epochs=0) # Warmup уже был, просто внедряем веса
    
    # 3. Тестовые промпты
    prompts = [
        "### Instruction:\nWrite a Python function to calculate fibonacci numbers\n\n### Response:",
        "### Instruction:\nWrite a short script to read a CSV file\n\n### Response:"
    ]
    
    print("\n📝 Результаты генерации:")
    print("=" * 70)
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # КЛЮЧЕВОЙ МОМЕНТ: правильные параметры generate для distilgpt2
        outputs = orch.model.generate(
            **inputs,
            max_new_tokens=80,            # Уменьшим, чтобы видеть суть
            do_sample=True,
            temperature=0.8,              # Чуть выше для разнообразия
            top_p=0.95,
            repetition_penalty=1.5,       # 🔑 КЛЮЧЕВОЙ ПАРАМЕТР: штраф за повторы
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        print("-" * 70)
    
    # 4. Очистка
    print("\n🧹 Eject и возврат в базовое состояние...")
    orch.eject_seed()
    print("✅ Готово!")

if __name__ == '__main__':
    main()
