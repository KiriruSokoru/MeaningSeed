# create_dataset.py
import json
import requests
import os

def download_alpaca_subset(output_path="data/alpaca_subset.json", num_samples=200):
    """Скачивает subset датасета Alpaca для анализа."""
    os.makedirs("data", exist_ok=True)
    
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"📥 Загрузка датасета Alpaca...")
    
    response = requests.get(url)
    data = response.json()
    
    # Берём первые num_samples примеров
    subset = data[:num_samples]
    
    # Форматируем в промпты
    prompts = []
    for item in subset:
        if item.get("input"):
            prompt = f"{item['instruction']}\n\nInput: {item['input']}"
        else:
            prompt = item['instruction']
        prompts.append(prompt)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Сохранено {len(prompts)} примеров в {output_path}")
    return prompts

if __name__ == "__main__":
    download_alpaca_subset()
