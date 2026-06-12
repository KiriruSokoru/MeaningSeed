import os
import json
import torch
import argparse
from rich.console import Console
from rich.prompt import Prompt

from meaning_seed.orchestrator import Orchestrator
from meaning_seed.model_adapter import get_model_adapter
from meaning_seed.i18n import get_t

console = Console()

def main():
    # 1. Выбор языка
    lang_choice = Prompt.ask("Выберите язык / Choose language [RU/EN]", choices=["RU", "EN", "ru", "en"], default="RU").lower()
    lang = "ru" if lang_choice == "ru" else "en"

    console.print(f"\n[bold cyan]{get_t('test_title', lang)}[/bold cyan]\n")
    
    seeds_dir = "./seeds"
    if not os.path.exists(seeds_dir):
        console.print(f"[red]{get_t('test_no_seeds', lang)}[/red]")
        return
        
    json_files = [f for f in os.listdir(seeds_dir) if f.endswith(".json")]
    if not json_files:
        console.print(f"[red]{get_t('test_no_seeds', lang)}[/red]")
        return

    seeds = []
    for filename in json_files:
        filepath = os.path.join(seeds_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                seeds.append({
                    "name": filename.replace(".json", ""),
                    "layer": data.get("layer_idx", "?"),
                    "scale": data.get("scale", "?"),
                    "count": len(data.get("master_indices", [])),
                    "model_type": data.get("model_type", "unknown"),
                    "path": filepath
                })
        except Exception:
            continue
    
    seeds.sort(key=lambda x: str(x["layer"]))
    
    console.print(f"[bold]{get_t('test_available', lang)}[/bold]")
    for i, seed in enumerate(seeds):
        console.print(f"  [{i+1}] {get_t('test_model', lang)}: {seed['model_type']}, Слой/Layer: {seed['layer']}, Нейронов/Neurons: {seed['count']}, x{seed['scale']} ({seed['name']})")
    
    choice = Prompt.ask(f"\n{get_t('test_select', lang)}", choices=[str(i) for i in range(1, len(seeds)+1)])
    selected = seeds[int(choice) - 1]

    # 2. Умный выбор модели на основе данных из семени
    console.print(f"\n[yellow]{get_t('test_seed_info', lang, model_type=selected['model_type'])}[/yellow]")
    
    # Если это qwen2, предлагаем 0.5B по умолчанию, но даем изменить
    if selected['model_type'] == 'qwen2':
        default_model = "Qwen/Qwen2.5-0.5B-Instruct"
    else:
        default_model = "unknown"
        
    model_id = Prompt.ask(get_t('test_model_prompt', lang), default=default_model)

    console.print(f"\n[yellow]{get_t('test_loading', lang)}[/yellow]")
    
    orchestrator = Orchestrator(model_name=model_id, device="auto", lang=lang)
    adapter = get_model_adapter(orchestrator.model, lang=lang)

    with open(selected["path"], 'r', encoding='utf-8') as f:
        seed_data = json.load(f)

    # Проверка совместимости перед применением
    if seed_data.get("model_type") != orchestrator.model.config.model_type:
        console.print(f"[bold red]❌ Ошибка: Семя создано для '{seed_data.get('model_type')}', а вы загружаете '{orchestrator.model.config.model_type}'[/bold red]")
        return

    adapter.scale_master_neurons(
        orchestrator.model,
        seed_data["layer_idx"],
        seed_data["master_indices"],
        seed_data["scale"]
    )

    console.print(f"[green]{get_t('test_applied', lang, layer=seed_data['layer_idx'], count=len(seed_data['master_indices']), scale=seed_data['scale'])}[/green]")
    console.print(f"\n[bold cyan]{get_t('test_chat', lang)}[/bold cyan]\n")
    
    orchestrator.model.eval()
    exit_words = {"выход", "exit", "quit", "q"}
    
    while True:
        try:
            prompt_text = Prompt.ask(f"\n[bold]{get_t('test_you', lang)}[/bold]")
            if prompt_text.lower().strip() in exit_words:
                break
            
            inputs = orchestrator.tokenizer(prompt_text, return_tensors="pt").to(orchestrator.model.device)
            
            with torch.no_grad():
                outputs = orchestrator.model.generate(
                    **inputs, max_new_tokens=100, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=orchestrator.tokenizer.eos_token_id
                )
            
            response = orchestrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_response = response[len(prompt_text):].strip()
            
            console.print(f"[bold green]{get_t('test_model', lang)}:[/bold green] {clean_response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Ошибка генерации: {e}[/red]")

    console.print(f"\n[bold]{get_t('test_ended', lang)}[/bold]")

if __name__ == "__main__":
    main()
