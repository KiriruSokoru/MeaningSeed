#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from meaning_seed.registry import load_seed, validate_seed_compatibility
from meaning_seed.orchestrator import Orchestrator
from meaning_seed.i18n import get_t

console = Console()
chat_console = Console(markup=False) 

# ЕДИНАЯ ПАПКА ДЛЯ МАСШТАБИРОВАННОЙ МОДЕЛИ
ACTIVE_SCALED_DIR = Path("./scaled_model")

def main():
    lang = Prompt.ask("Choose language / Выберите язык (ru/en)", choices=["ru", "en"], default="ru")
    
    console.print(Panel("[bold cyan]MeaningSeed: Universal Seed Tester[/bold cyan]", expand=False))
    
    seed_input = Prompt.ask(
        get_t("seed_path_prompt", lang, default="./seeds/"),
        default="./seeds"
    ).strip()
    
    seed_path = Path(seed_input)
    if seed_path.is_dir():
        seeds = list(seed_path.glob("*.json"))
        if not seeds:
            console.print(f"[red]{get_t('no_seeds_found', lang)}[/red]")
            return
        
        console.print(f"\n[bold]{get_t('available_seeds', lang)}[/bold]")
        table = Table(show_header=True, box=None)
        table.add_column("#", style="cyan")
        table.add_column(get_t('seed_name', lang), style="bold")
        table.add_column(get_t('target_model', lang), style="dim")
        
        for i, s in enumerate(seeds, 1):
            data = load_seed(s)
            model_name = data.get("model_name", "base_model")
            table.add_row(str(i), s.name, model_name)
        console.print(table)
        
        choice = Prompt.ask(get_t('select_seed', lang), default="1")
        try:
            seed_path = seeds[int(choice) - 1]
        except (ValueError, IndexError):
            console.print(f"[red]{get_t('invalid_choice', lang)}[/red]")
            return

    try:
        seed_data = load_seed(str(seed_path))
    except Exception as e:
        console.print(f"[red]{get_t('seed_read_error', lang, error=str(e))}[/red]")
        return

    model_name = seed_data.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    scaling_factor = seed_data.get("scaling_factor", seed_data.get("scale", 1.0))
    
    console.print(f"\n[green]{get_t('seed_loaded', lang)}[/green] {seed_path.name}")
    console.print(f"  {get_t('target_model', lang)}: [bold]{model_name}[/bold]")
    console.print(f"  {get_t('scaling_factor', lang)}: [bold]x{scaling_factor}[/bold]")

    # Проверяем наличие единой папки scaled_model
    use_scaled_dir = ACTIVE_SCALED_DIR.exists() and (ACTIVE_SCALED_DIR / "config.json").exists()
    
    if use_scaled_dir:
        console.print(f"\n[green]✓ Найдена активная масштабированная модель в:[/green] {ACTIVE_SCALED_DIR}")
        if Confirm.ask("Использовать её? (Рекомендуется)", default=True):
            load_target = str(ACTIVE_SCALED_DIR)
            load_mode = "scaled"
        else:
            load_target = model_name
            load_mode = "fresh"
    else:
        console.print(f"\n[dim]Папка {ACTIVE_SCALED_DIR} не найдена. Будет загружена базовая модель и применен сид.[/dim]")
        load_target = model_name
        load_mode = "fresh"

    custom_model = Prompt.ask(
        f"\nПереопределить путь к модели? (Enter для: {load_target})",
        default=""
    ).strip()
    
    if custom_model:
        load_target = custom_model
        load_mode = "fresh"

    console.print(f"\n[bold]Загрузка модели:[/bold] {load_target} ...")
    try:
        orchestrator = Orchestrator(model_name=load_target, lang=lang)
    except Exception as e:
        console.print(f"[red]Не удалось загрузить модель: {e}[/red]")
        return

    if load_mode == "fresh":
        console.print("[bold]Проверка совместимости сида...[/bold]")
        is_compatible, error_msg = validate_seed_compatibility(seed_data, orchestrator.model.config)
        
        if not is_compatible:
            console.print(f"[red]Ошибка совместимости:[/red] {error_msg}")
            return
        
        console.print("[green]Совместимость подтверждена. Применение сида...[/green]")
        try:
            orchestrator.load_and_apply_seed(str(seed_path))
        except Exception as e:
            console.print(f"[red]Ошибка применения сида: {e}[/red]")
            return
    else:
        console.print("[green]Загружена готовая масштабированная модель.[/green]")

    console.print("\n" + "="*50)
    console.print("[bold cyan]Интерактивный цикл генерации[/bold cyan]")
    console.print("[dim]Команды: 'exit' (выход), 'clear' (очистить историю)[/dim]")
    console.print("="*50 + "\n")
    
    history = []
    while True:
        try:
            user_input = input("Вы: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q', 'выход']:
                console.print("\n[yellow]Завершение работы.[/yellow]")
                break
            if user_input.lower() == 'clear':
                history = []
                chat_console.print("[dim]История очищена.[/dim]\n")
                continue

            history.append({"role": "user", "content": user_input})
            chat_console.print("\nМодель: (генерация...)", end="\r")
            
            try:
                response = orchestrator.generate(
                    messages=history,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            except Exception as e:
                chat_console.print(f"\n[red]Ошибка генерации: {e}[/red]")
                if len(history) >= 2:
                    history = history[:-2]
                continue

            chat_console.print(" " * 60, end="\r")
            history.append({"role": "assistant", "content": response})
            chat_console.print(f"[bold blue]Модель:[/bold blue]\n{response}\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Прервано.[/yellow]")
            break
        except Exception as e:
            chat_console.print(f"\n[red]Критическая ошибка: {e}[/red]")
            if len(history) >= 2:
                history = history[:-2]

if __name__ == "__main__":
    main()
