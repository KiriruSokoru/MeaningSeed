import os
import json
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from meaning_seed.orchestrator import Orchestrator
from meaning_seed.model_adapter import get_model_adapter

console = Console()

def list_available_seeds():
    """Показывает доступные семена в папке seeds/"""
    seeds_dir = "./seeds"
    if not os.path.exists(seeds_dir):
        return []
    
    seeds = []
    for filename in os.listdir(seeds_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(seeds_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    seeds.append({
                        "name": filename.replace(".json", ""),
                        "model_type": data.get("model_type", "unknown"),
                        "layer": data.get("layer_idx", "?"),
                        "scale": data.get("scale", "?"),
                        "neurons_count": len(data.get("master_indices", [])),
                        "path": filepath
                    })
                except Exception:
                    continue
    return sorted(seeds, key=lambda x: x["layer"])

def main():
    console.print(Panel.fit(
        "[bold cyan]🔑 MeaningSeed: Мгновенное применение семени[/bold cyan]\n"
        "[dim]Загрузка чистой модели и хирургическое применение Real-World Proof[/dim]",
        border_style="cyan"
    ))

    # 1. Выбор семени
    seeds = list_available_seeds()
    if not seeds:
        console.print("[bold red]❌ В папке ./seeds/ не найдено ни одного семени.[/bold red]")
        console.print("Сначала запустите main.py для создания семян.")
        return

    console.print("\n[bold]Доступные семена:[/bold]")
    for i, seed in enumerate(seeds):
        console.print(f"  [{i+1}] Слой {seed['layer']}: {seed['neurons_count']} нейронов, масштаб x{seed['scale']} ({seed['name']})")
    
    choice = Prompt.ask("\nВыберите номер семени для применения", choices=[str(i) for i in range(1, len(seeds)+1)])
    selected_seed = seeds[int(choice) - 1]

    # 2. Загрузка чистой модели
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    console.print(f"\n[bold yellow]⏳ Загрузка чистой модели {model_id}...[/bold yellow]")
    orchestrator = Orchestrator(model_name=model_id, device="auto")
    adapter = get_model_adapter(orchestrator.model)

    # 3. Применение семени
    console.print(f"\n[bold green]⚡ Применение семени '{selected_seed['name']}'...[/bold green]")
    
    with open(selected_seed["path"], 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
        
    layer_idx = seed_data["layer_idx"]
    master_indices = seed_data["master_indices"]
    scale = seed_data["scale"]
    
    # Хирургическое применение через адаптер
    adapter.scale_master_neurons(orchestrator.model, layer_idx, master_indices, scale)
    
    console.print(Panel(
        f"[green]✅ Успешно![/green]\n"
        f"• Слой: {layer_idx}\n"
        f"• Нейронов изменено: {len(master_indices)}\n"
        f"• Масштаб: x{scale}\n"
        f"• Модель теперь 'настроена' на паттерн этого семени.",
        border_style="green"
    ))

    # 4. Интерактивный чат с модифицированной моделью
    console.print("\n[bold cyan]💬 Интерактивный режим (введите 'выход' для завершения)[/bold cyan]")
    orchestrator.model.eval()
    
    while True:
        prompt_text = Prompt.ask("\n[bold]Вы[/bold]")
        if prompt_text.lower() in ["выход", "exit", "quit"]:
            break
            
        inputs = orchestrator.tokenizer(prompt_text, return_tensors="pt").to(orchestrator.model.device)
        
        with torch.no_grad():
            outputs = orchestrator.model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
                pad_token_id=orchestrator.tokenizer.eos_token_id,
                eos_token_id=orchestrator.tokenizer.eos_token_id
            )
            
        response = orchestrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Убираем промпт из ответа для чистоты
        clean_response = response[len(prompt_text):].strip()
        
        console.print(f"[bold green]Модель[/bold green]: {clean_response}")

if __name__ == "__main__":
    main()
