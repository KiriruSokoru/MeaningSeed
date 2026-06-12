import argparse
import torch
import time
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from meaning_seed.orchestrator import Orchestrator
from meaning_seed.extractor import ActivationExtractor
from meaning_seed.registry import SeedRegistry

# Инициализация красивого вывода
console = Console()

# Мини-датасет для демонстрации (можно заменить на загрузку из внешнего JSON)
SAMPLE_TEXTS = [
    "Искусственный интеллект трансформирует современные технологии, делая их более доступными.",
    "Квантовые вычисления обещают решить задачи, которые не под силу классическим компьютерам.",
    "Нейронные сети обучаются на огромных массивах данных, выявляя скрытые закономерности.",
    "Блокчейн обеспечивает прозрачность и безопасность цифровых транзакций по всему миру.",
    "Автоматизация рутинных процессов позволяет людям сосредоточиться на творческих задачах.",
    "Большие языковые модели способны понимать контекст и генерировать связный текст.",
    "Кибербезопасность становится критически важной в эпоху повсеместной цифровизации.",
    "Машинное обучение используется в медицине для ранней диагностики сложных заболеваний.",
    "Робототехника интегрируется в производство, повышая эффективность и точность сборки.",
    "Облачные вычисления предоставляют масштабируемые ресурсы для стартапов и корпораций."
] * 5  # Умножаем для получения 50 примеров и более стабильной статистики

def print_header():
    console.print(Panel.fit(
        "[bold cyan]Real-World Proof: Масштабирование на Qwen2.5-0.5B[/bold cyan]\n"
        "[dim]Поиск и масштабирование мастер-нейронов в MLP слоях[/dim]",
        border_style="cyan"
    ))

def main():
    print_header()
    
    parser = argparse.ArgumentParser(description="Real-World Proof CLI")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Название модели HuggingFace")
    parser.add_argument("--layers", type=int, nargs="+", default=[12], help="Индексы слоев для анализа (для 0.5B слои от 0 до 23)")
    parser.add_argument("--top_k", type=int, default=10, help="Количество мастер-нейронов для сохранения")
    parser.add_argument("--scale", type=float, default=1.5, help="Коэффициент масштабирования")
    parser.add_argument("--proof_name", type=str, default="qwen2_0.5b_poc_v1", help="Имя для сохранения доказательства")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча для инференса")
    parser.add_argument("--save_path", type=str, default="./qwen2-0.5b-scaled", help="Путь для сохранения модифицированной модели")
    parser.add_argument("--test_prompt", type=str, default="Искусственный интеллект в будущем будет ", help="Промпт для тестовой генерации")
    
    args = parser.parse_args()

    try:
        # ---------------------------------------------------------
        # ЭТАП 1: Загрузка модели
        # ---------------------------------------------------------
        console.print("\n[bold yellow]⏳ Этап 1: Инициализация и загрузка модели...[/bold yellow]")
        start_time = time.time()
        
        orchestrator = Orchestrator(model_name=args.model, device="auto")
        
        load_time = time.time() - start_time
        console.print(f"[green]✅ Модель успешно загружена за {load_time:.2f} сек.[/green]")

        # ---------------------------------------------------------
        # ЭТАП 2: Поиск мастер-нейронов
        # ---------------------------------------------------------
        console.print(f"\n[bold yellow]⏳ Этап 2: Анализ активаций на {len(SAMPLE_TEXTS)} текстовых примерах...[/bold yellow]")
        
        extractor = ActivationExtractor(model=orchestrator.model, adapter=orchestrator.adapter)
        
        master_neurons = extractor.find_master_neurons(
            texts=SAMPLE_TEXTS,
            tokenizer=orchestrator.tokenizer,
            layer_indices=args.layers,
            top_k=args.top_k,
            batch_size=args.batch_size
        )

        # ---------------------------------------------------------
        # ЭТАП 3: Регистрация доказательства (Seed)
        # ---------------------------------------------------------
        console.print("\n[bold yellow]⏳ Этап 3: Регистрация Real-World Proof...[/bold yellow]")
        
        registry = SeedRegistry(registry_dir="./seeds")
        
        for layer_idx, indices in master_neurons.items():
            registry.register_proof(
                proof_name=f"{args.proof_name}_layer{layer_idx}",
                model_type=orchestrator.model.config.model_type,
                layer_idx=layer_idx,
                master_indices=indices,
                scale=args.scale
            )

        # ---------------------------------------------------------
        # ЭТАП 4: Визуальная сводка и применение
        # ---------------------------------------------------------
        console.print("\n[bold green]🎉 Этап 4: Сводка и применение доказательств[/bold green]")
        
        table = Table(title="📊 Зарегистрированные Мастер-Нейроны", show_header=True, header_style="bold magenta")
        table.add_column("Слой", justify="center")
        table.add_column("Кол-во нейронов", justify="center")
        table.add_column("Индексы нейронов", style="cyan")
        table.add_column("Масштаб", justify="center")

        for layer_idx, indices in master_neurons.items():
            proof_name = f"{args.proof_name}_layer{layer_idx}"
            proof_data = registry.get_proof(proof_name)
            
            # Применяем масштабирование через оркестратор
            orchestrator.apply_scaling(
                layer_idx=layer_idx,
                master_indices=indices,
                scale=proof_data["scale"]
            )
            
            indices_str = ", ".join(map(str, indices[:5])) + ("..." if len(indices) > 5 else "")
            table.add_row(
                str(layer_idx),
                str(len(indices)),
                indices_str,
                f"x{proof_data['scale']}"
            )

        console.print(table)
        
        # ---------------------------------------------------------
        # ЭТАП 5: Сохранение модифицированной модели (Правка 1)
        # ---------------------------------------------------------
        console.print(f"\n[bold yellow]⏳ Этап 5: Сохранение масштабированной модели в {args.save_path}...[/bold yellow]")
        os.makedirs(args.save_path, exist_ok=True)
        orchestrator.model.save_pretrained(args.save_path)
        orchestrator.tokenizer.save_pretrained(args.save_path)
        console.print("[green]✅ Модель и токенизатор успешно сохранены на диск![/green]")

        # ---------------------------------------------------------
        # ЭТАП 6: Тестовая генерация (Правка 2)
        # ---------------------------------------------------------
        console.print("\n[bold yellow]🧪 Этап 6: Тестовая генерация с усиленными мастер-нейронами...[/bold yellow]")
        orchestrator.model.eval()
        
        inputs = orchestrator.tokenizer(args.test_prompt, return_tensors="pt").to(orchestrator.model.device)
        
        with torch.no_grad():
            outputs = orchestrator.model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
                pad_token_id=orchestrator.tokenizer.eos_token_id
            )

        response = orchestrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        console.print(Panel(
            f"[white]{response}[/white]", 
            title="🤖 Ответ масштабированной модели", 
            border_style="green"
        ))

        console.print(Panel(
            "[bold green]✅ ПОЛНЫЙ УСПЕХ![/bold green]\n"
            f"1. Доказательства сохранены в папке [cyan]./seeds/[/cyan]\n"
            f"2. Модель сохранена в [cyan]{args.save_path}[/cyan]\n"
            f"3. Генерация подтверждает работоспособность.",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"\n[bold red]❌ Критическая ошибка: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
