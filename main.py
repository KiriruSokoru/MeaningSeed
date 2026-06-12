import os
import sys
import json
import time
import torch
import psutil
import requests
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.markup import escape
from rich.markdown import Markdown
from rich.markup import escape

from meaning_seed.orchestrator import Orchestrator
from meaning_seed.extractor import ActivationExtractor
from meaning_seed.registry import SeedRegistry
from meaning_seed.model_adapter import get_model_adapter
from transformers import AutoConfig

console = Console()

# ============================================================
# СИСТЕМА ЛОКАЛИЗАЦИИ (RU/ENG)
# ============================================================
I18N = {
    "ru": {
        "welcome_title": "🧠 MeaningSeed: Real-World Proof",
        "welcome_subtitle": "Хирургия нейронных сетей через топологические семена",
        "choose_language": "Выберите язык / Choose language",
        
        "about_title": "О ПРОЕКТЕ",
        "about_text": """[bold cyan]MeaningSeed[/bold cyan] — это инструмент для выделения, консервации и применения "мастер-нейронов" в больших языковых моделях.

[bold yellow]Идея:[/bold yellow] В каждой LLM существуют специфические нейроны, которые несут ключевую семантическую нагрузку для конкретных задач. Мы называем их "мастер-нейронами". Находя их и сохраняя в виде легких JSON-семян, мы получаем возможность мгновенно "настраивать" любую модель на нужный паттерн без переобучения.

[bold yellow]Математическая основа:[/bold yellow] Проект опирается на [bold]Гипотезу Сокола[/bold] — предположение о том, что аттрактор задачи в пространстве LLM формируется через нейроны с положительной кривизной Риччи. Применяемая техника масштабирования весов — это "[bold]Хирургия Перельмана[/bold]", отсылка к методу доказательства гипотезы Пуанкаре через поток Риччи.

[bold yellow]Ключевые концепции:[/bold yellow]
  • [cyan]Топологический стартер[/cyan] — инициализация мастер-нейронов
  • [cyan]Targeted Warmup[/cyan] — направленный прогрев для стабилизации
  • [cyan]Incremental Base Weight Backup[/cyan] — безопасное резервное копирование

[bold yellow]Контакты автора:[/bold yellow]
  📱 Telegram: [link=https://t.me/Kirill_Sokol1]@Kirill_Sokol1[/link]
  📧 Email: [link=mailto:kiriru_sokoru@proton.me]kiriru_sokoru@proton.me[/link]
  💬 Баги, идеи, сотрудничество — пишите!""",
        
        "process_title": "ЧТО МЫ СЕЙЧАС БУДЕМ ДЕЛАТЬ",
        "process_text": """[bold]Пайплайн состоит из 4 этапов:[/bold]

  [cyan]1.[/cyan] [bold]Загрузка модели[/bold] — скачиваем LLM с HuggingFace
  [cyan]2.[/cyan] [bold]Сбор активаций[/bold] — прогоняем датасет через модель и замеряем отклик каждого нейрона
  [cyan]3.[/cyan] [bold]Выделение мастер-нейронов[/bold] — находим top-K самых активных нейронов в каждом слое
  [cyan]4.[/cyan] [bold]Консервация[/bold] — сохраняем их индексы в JSON-семя (Real-World Proof)

После этого семя можно мгновенно применить к любой совместимой модели.""",

        "step_model": "ШАГ 1: ВЫБОР МОДЕЛИ",
        "step_dataset": "ШАГ 2: ВЫБОР ДАТАСЕТА",
        "step_depth": "ШАГ 3: ГЛУБИНА АНАЛИЗА",
        "step_intensity": "ШАГ 4: ИНТЕНСИВНОСТЬ МАСШТАБИРОВАНИЯ",
        "step_prompt": "ШАГ 5: ТЕСТОВЫЙ ПРОМПТ",
        
        "model_options": """  [1] [bold green]Qwen2.5-0.5B[/bold green] (рекомендуется для CPU, ~1GB RAM)
  [2] [bold yellow]Qwen2.5-1.5B[/bold yellow] (требуется ~3GB свободной RAM)
  [3] [bold yellow]Qwen2.5-3B[/bold yellow] (требуется ~6GB свободной RAM)
  [4] [bold red]Своя модель с HuggingFace[/bold red] (укажите ID)""",
        "model_choice": "Ваш выбор",
        "model_custom_id": "Введите ID модели (например, meta-llama/Llama-3.2-1B-Instruct)",
        
        "ram_warning": "[bold red]⚠️ ВНИМАНИЕ:[/bold red] У вас {available:.1f}GB свободной RAM, а модель требует ~{required:.1f}GB. Система может уйти в swap и работать ОЧЕНЬ медленно.",
        "ram_ok": "[green]✓ Достаточно RAM: свободно {available:.1f}GB, требуется ~{required:.1f}GB[/green]",
        "ram_check": "Проверка системных ресурсов...",
        "ram_proceed_anyway": "Всё равно продолжить?",
        
        "dataset_options": """  [1] Быстрый демо-набор (50 фраз, ~1 минута)
  [2] [bold]Реальный набор Alpaca[/bold] (200 инструкций, ~15-20 минут)
  [3] Свой JSON файл""",
        "dataset_choice": "Ваш выбор",
        "dataset_custom_path": "Введите путь к JSON файлу",
        "dataset_loading": "📥 Скачивание датасета Alpaca...",
        "dataset_loaded": "✅ Датасет загружен",
        
        "depth_options": """  [1] Быстрый тест (один средний слой)
  [2] [bold]Полный анализ[/bold] (все слои модели — даст полную карту)
  [3] Выбранные слои (введите через пробел, например: 10 12 14)""",
        "depth_choice": "Ваш выбор",
        "depth_custom": "Введите номера слоев через пробел",
        
        "intensity_options": """  [1] [bold green]Сбалансированная[/bold green] (x1.5, топ-10 нейронов) — рекомендуется
  [2] [bold yellow]Агрессивная[/bold yellow] (x2.0, топ-20 нейронов) — возможны галлюцинации
  [3] [bold cyan]Подавление[/bold cyan] (x0.5, топ-10) — делает ответы шаблоннее""",
        "intensity_choice": "Ваш выбор",
        
        "prompt_default": "Главное преимущество нейросетей заключается в ",
        "prompt_ask": "Введите фразу для продолжения",
        
        "stage_loading": "⏳ Загрузка модели...",
        "stage_analysis": "⏳ Анализ активаций ({count} примеров, batch={batch})...",
        "stage_register": "⏳ Регистрация и применение модификаций...",
        "stage_save": "⏳ Сохранение модели в {path}...",
        "stage_test": "🧪 Тестовая генерация...",
        
        "success_title": "🎉 ЭКСПЕРИМЕНТ ЗАВЕРШЕН",
        "success_text": """• Семена сохранены в [cyan]./seeds/[/cyan]
• Модель сохранена в [cyan]{path}[/cyan]
• Количество семян: [bold]{count}[/bold]""",
        
        "menu_title": "ЧТО ДАЛЬШЕ?",
        "menu_options": """  [1] [bold green]🧪 Протестировать созданные семена[/bold green] (интерактивный чат)
  [2] [bold cyan]🔄 Повторить эксперимент[/bold green] (с другими параметрами)
  [3] [bold red]🚪 Завершить работу[/bold red]""",
        "menu_choice": "Ваш выбор",
        
        "test_title": "🔑 Применение семени",
        "test_select": "Выберите номер семени",
        "test_applying": "⚡ Применение семени...",
        "test_applied": "✅ Семя применено. Слой {layer}, {count} нейронов, масштаб x{scale}",
        "test_chat": "💬 Интерактивный чат (введите 'выход' для завершения)",
        "test_you": "Вы",
        "test_model": "Модель",
        
        "error_no_seeds": "❌ В папке ./seeds/ нет семян. Сначала выполните генерацию.",
        "error_critical": "❌ Критическая ошибка: {error}",
        "error_interrupted": "⏹️ Прервано пользователем",
        
        "goodbye": "👋 Спасибо за использование MeaningSeed! До новых экспериментов.",
    },
    "en": {
        "welcome_title": "🧠 MeaningSeed: Real-World Proof",
        "welcome_subtitle": "Neural network surgery through topological seeds",
        "choose_language": "Выберите язык / Choose language",
        
        "about_title": "ABOUT THE PROJECT",
        "about_text": """[bold cyan]MeaningSeed[/bold cyan] is a tool for extracting, preserving, and applying "master neurons" in large language models.

[bold yellow]The Idea:[/bold yellow] Every LLM contains specific neurons that carry key semantic load for particular tasks. We call them "master neurons". By finding them and saving as lightweight JSON seeds, we can instantly "tune" any model to a desired pattern without retraining.

[bold yellow]Mathematical Foundation:[/bold yellow] The project is based on the [bold]Sokol Hypothesis[/bold] — the assumption that a task attractor in LLM space forms through neurons with positive Ricci curvature. The weight scaling technique applied is "[bold]Perelman Surgery[/bold]", referencing the proof of the Poincaré conjecture via Ricci flow.

[bold yellow]Key Concepts:[/bold yellow]
  • [cyan]Topological Starter[/cyan] — initialization of master neurons
  • [cyan]Targeted Warmup[/cyan] — directed warmup for stabilization
  • [cyan]Incremental Base Weight Backup[/cyan] — safe weight backup

[bold yellow]Author Contacts:[/bold yellow]
  📱 Telegram: [link=https://t.me/Kirill_Sokol1]@Kirill_Sokol1[/link]
  📧 Email: [link=mailto:kiriru_sokoru@proton.me]kiriru_sokoru@proton.me[/link]
  💬 Bugs, ideas, collaboration — feel free to reach out!""",
        
        "process_title": "WHAT WE'RE GOING TO DO",
        "process_text": """[bold]The pipeline has 4 stages:[/bold]

  [cyan]1.[/cyan] [bold]Load model[/bold] — download LLM from HuggingFace
  [cyan]2.[/cyan] [bold]Collect activations[/bold] — run dataset through the model and measure each neuron's response
  [cyan]3.[/cyan] [bold]Extract master neurons[/bold] — find top-K most active neurons in each layer
  [cyan]4.[/cyan] [bold]Preserve[/bold] — save their indices to a JSON seed (Real-World Proof)

After this, the seed can be instantly applied to any compatible model.""",

        "step_model": "STEP 1: CHOOSE MODEL",
        "step_dataset": "STEP 2: CHOOSE DATASET",
        "step_depth": "STEP 3: ANALYSIS DEPTH",
        "step_intensity": "STEP 4: SCALING INTENSITY",
        "step_prompt": "STEP 5: TEST PROMPT",
        
        "model_options": """  [1] [bold green]Qwen2.5-0.5B[/bold green] (recommended for CPU, ~1GB RAM)
  [2] [bold yellow]Qwen2.5-1.5B[/bold yellow] (requires ~3GB free RAM)
  [3] [bold yellow]Qwen2.5-3B[/bold yellow] (requires ~6GB free RAM)
  [4] [bold red]Custom model from HuggingFace[/bold red] (specify ID)""",
        "model_choice": "Your choice",
        "model_custom_id": "Enter model ID (e.g., meta-llama/Llama-3.2-1B-Instruct)",
        
        "ram_warning": "[bold red]⚠️ WARNING:[/bold red] You have {available:.1f}GB free RAM, but the model needs ~{required:.1f}GB. System may swap and work VERY slowly.",
        "ram_ok": "[green]✓ RAM is sufficient: {available:.1f}GB free, ~{required:.1f}GB required[/green]",
        "ram_check": "Checking system resources...",
        "ram_proceed_anyway": "Proceed anyway?",
        
        "dataset_options": """  [1] Quick demo set (50 phrases, ~1 minute)
  [2] [bold]Real Alpaca dataset[/bold] (200 instructions, ~15-20 minutes)
  [3] Custom JSON file""",
        "dataset_choice": "Your choice",
        "dataset_custom_path": "Enter path to JSON file",
        "dataset_loading": "📥 Downloading Alpaca dataset...",
        "dataset_loaded": "✅ Dataset loaded",
        
        "depth_options": """  [1] Quick test (one middle layer)
  [2] [bold]Full analysis[/bold] (all model layers — gives complete map)
  [3] Custom layers (enter space-separated, e.g.: 10 12 14)""",
        "depth_choice": "Your choice",
        "depth_custom": "Enter layer numbers separated by spaces",
        
        "intensity_options": """  [1] [bold green]Balanced[/bold green] (x1.5, top-10 neurons) — recommended
  [2] [bold yellow]Aggressive[/bold yellow] (x2.0, top-20 neurons) — may cause hallucinations
  [3] [bold cyan]Suppression[/bold cyan] (x0.5, top-10) — makes responses more template-like""",
        "intensity_choice": "Your choice",
        
        "prompt_default": "The main advantage of neural networks is ",
        "prompt_ask": "Enter a phrase to continue",
        
        "stage_loading": "⏳ Loading model...",
        "stage_analysis": "⏳ Analyzing activations ({count} examples, batch={batch})...",
        "stage_register": "⏳ Registering and applying modifications...",
        "stage_save": "⏳ Saving model to {path}...",
        "stage_test": "🧪 Test generation...",
        
        "success_title": "🎉 EXPERIMENT COMPLETED",
        "success_text": """• Seeds saved to [cyan]./seeds/[/cyan]
• Model saved to [cyan]{path}[/cyan]
• Number of seeds: [bold]{count}[/bold]""",
        
        "menu_title": "WHAT'S NEXT?",
        "menu_options": """  [1] [bold green]🧪 Test created seeds[/bold green] (interactive chat)
  [2] [bold cyan]🔄 Repeat experiment[/bold green] (with different parameters)
  [3] [bold red]🚪 Exit[/bold red]""",
        "menu_choice": "Your choice",
        
        "test_title": "🔑 Applying seed",
        "test_select": "Select seed number",
        "test_applying": "⚡ Applying seed...",
        "test_applied": "✅ Seed applied. Layer {layer}, {count} neurons, scale x{scale}",
        "test_chat": "💬 Interactive chat (type 'exit' to quit)",
        "test_you": "You",
        "test_model": "Model",
        
        "error_no_seeds": "❌ No seeds in ./seeds/. Please run generation first.",
        "error_critical": "❌ Critical error: {error}",
        "error_interrupted": "⏹️ Interrupted by user",
        
        "goodbye": "👋 Thank you for using MeaningSeed! Until next experiments.",
    }
}

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_t(key, lang="ru", **kwargs):
    """Получить переведенную строку"""
    text = I18N.get(lang, I18N["ru"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text

def get_available_ram_gb():
    """Получить доступную RAM в GB"""
    try:
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 4.0  # fallback

def estimate_model_ram_gb(model_id):
    """Оценить требуемую RAM для модели"""
    try:
        config = AutoConfig.from_pretrained(model_id)
        # Примерная формула: 2GB на 1B параметров (float16) + 1GB overhead
        num_params = getattr(config, "num_hidden_layers", 24) * \
                     getattr(config, "hidden_size", 896) ** 2 * 12
        params_billions = num_params / 1e9
        return max(1.0, params_billions * 2 + 1.0)
    except Exception:
        return 2.0  # fallback для неизвестных моделей

def ensure_alpaca_dataset(lang="ru"):
    """Скачать датасет Alpaca, если его нет"""
    base_dir = Path(__file__).parent.resolve()
    data_dir = base_dir / "data"
    target_path = data_dir / "alpaca_subset.json"
    
    if target_path.exists():
        return str(target_path)
    
    console.print(f"\n[yellow]{get_t('dataset_loading', lang)}[/yellow]")
    data_dir.mkdir(exist_ok=True)
    
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    try:
        response = requests.get(url, timeout=60)
        data = response.json()[:200]
        
        prompts = []
        for item in data:
            if item.get("input"):
                prompts.append(f"{item['instruction']}\n\nInput: {item['input']}")
            else:
                prompts.append(item['instruction'])
        
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]{get_t('dataset_loaded', lang)}[/green]")
        return str(target_path)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return None

def check_ram_safety(required_gb, lang="ru"):
    """Проверить, хватит ли RAM"""
    available = get_available_ram_gb()
    console.print(f"\n[cyan]{get_t('ram_check', lang)}[/cyan]")
    
    if available >= required_gb:
        console.print(get_t('ram_ok', lang, available=available, required=required_gb))
        return True
    else:
        console.print(get_t('ram_warning', lang, available=available, required=required_gb))
        return Confirm.ask(get_t('ram_proceed_anyway', lang), default=False)

def list_seeds():
    """Список доступных семян"""
    seeds_dir = "./seeds"
    if not os.path.exists(seeds_dir):
        return []
    
    seeds = []
    for filename in os.listdir(seeds_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(seeds_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    seeds.append({
                        "name": filename.replace(".json", ""),
                        "layer": data.get("layer_idx", "?"),
                        "scale": data.get("scale", "?"),
                        "count": len(data.get("master_indices", [])),
                        "path": filepath
                    })
            except Exception:
                continue
    return sorted(seeds, key=lambda x: x["layer"])

# ============================================================
# ИНТЕРАКТИВНЫЕ МЕНЮ
# ============================================================

def choose_language():
    """Выбор языка"""
    console.print(Panel.fit(
        f"[bold cyan]{get_t('welcome_title', 'ru')}[/bold cyan]\n"
        f"[dim]{get_t('welcome_subtitle', 'ru')}[/dim]\n\n"
        f"[bold cyan]{get_t('welcome_title', 'en')}[/bold cyan]\n"
        f"[dim]{get_t('welcome_subtitle', 'en')}[/dim]",
        border_style="cyan"
    ))
    choice = Prompt.ask(f"\n{get_t('choose_language', 'ru')} [RU/EN]", choices=["RU", "EN", "ru", "en"], default="EN").lower()
    return "ru" if choice == "ru" else "en"

def show_onboarding(lang):
    """Показать онбординг"""
    console.print(Panel(
        get_t('about_text', lang),
        title=f"[bold]{get_t('about_title', lang)}[/bold]",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()
    console.print(Panel(
        get_t('process_text', lang),
        title=f"[bold]{get_t('process_title', lang)}[/bold]",
        border_style="yellow",
        padding=(1, 2)
    ))
    Prompt.ask("\n[bold]Press Enter to continue...[/bold]")

def choose_model(lang):
    """Выбор модели с проверкой RAM"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_t('step_model', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('model_options', lang)))
    
    choice = Prompt.ask(get_t('model_choice', lang), choices=["1", "2", "3", "4"], default="1")
    
    model_map = {
        "1": ("Qwen/Qwen2.5-0.5B-Instruct", 1.0),
        "2": ("Qwen/Qwen2.5-1.5B-Instruct", 3.0),
        "3": ("Qwen/Qwen2.5-3B-Instruct", 6.0),
    }
    
    if choice in model_map:
        model_id, required = model_map[choice]
    else:
        model_id = Prompt.ask(get_t('model_custom_id', lang))
        required = estimate_model_ram_gb(model_id)
    
    if not check_ram_safety(required, lang):
        return None
    
    return model_id

def choose_dataset(lang):
    """Выбор датасета"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_t('step_dataset', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('dataset_options', lang)))
    
    choice = Prompt.ask(get_t('dataset_choice', lang), choices=["1", "2", "3"], default="2")
    
    if choice == "1":
        return None  # встроенный датасет
    elif choice == "2":
        return ensure_alpaca_dataset(lang)
    else:
        path = Prompt.ask(get_t('dataset_custom_path', lang))
        return os.path.abspath(path) if os.path.exists(path) else None

def choose_depth(lang, model_id):
    """Выбор глубины анализа"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_t('step_depth', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('depth_options', lang)))
    
    choice = Prompt.ask(get_t('depth_choice', lang), choices=["1", "2", "3"], default="1")
    
    if choice == "1":
        return [12]
    elif choice == "2":
        try:
            cfg = AutoConfig.from_pretrained(model_id)
            num_layers = getattr(cfg, "num_hidden_layers", 24)
            return list(range(num_layers))
        except Exception:
            return list(range(24))
    else:
        layers_str = Prompt.ask(get_t('depth_custom', lang), default="10 12 14")
        try:
            return [int(x.strip()) for x in layers_str.split()]
        except Exception:
            return [12]

def choose_intensity(lang):
    """Выбор интенсивности"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_t('step_intensity', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('intensity_options', lang)))
    
    choice = Prompt.ask(get_t('intensity_choice', lang), choices=["1", "2", "3"], default="1")
    
    intensity_map = {
        "1": (1.5, 10),
        "2": (2.0, 20),
        "3": (0.5, 10),
    }
    return intensity_map[choice]

def choose_prompt(lang):
    """Выбор тестового промпта"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_t('step_prompt', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    return Prompt.ask(get_t('prompt_ask', lang), default=get_t('prompt_default', lang))

# ============================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================

def run_experiment(config, lang):
    """Запустить эксперимент"""
    try:
        # Загрузка модели
        console.print(f"\n[bold yellow]{get_t('stage_loading', lang)}[/bold yellow]")
        start_time = time.time()
        orchestrator = Orchestrator(model_name=config["model_id"], device="auto", lang=lang)
        console.print(f"[green]✅ {time.time() - start_time:.2f}s[/green]")
        
        # Загрузка текстов
        if config["dataset_path"]:
            with open(config["dataset_path"], 'r', encoding='utf-8') as f:
                texts = json.load(f)
        else:
            texts = [
                "Neural networks learn from large datasets to find hidden patterns.",
                "Machine learning is used in medicine for early disease diagnosis.",
                "Artificial intelligence transforms modern technologies.",
                "Quantum computing promises to solve complex problems.",
                "Blockchain ensures transparency of digital transactions."
            ] * 10
        
        # Анализ активаций
        console.print(f"\n[bold yellow]{get_t('stage_analysis', lang, count=len(texts), batch=config['batch_size'])}[/bold yellow]")
        extractor = ActivationExtractor(model=orchestrator.model, adapter=orchestrator.adapter, lang=lang)
        
        master_neurons = extractor.find_master_neurons(
            texts=texts,
            tokenizer=orchestrator.tokenizer,
            layer_indices=config["layers"],
            top_k=config["top_k"],
            batch_size=config["batch_size"]
        )
        
        # Регистрация
        console.print(f"\n[bold yellow]{get_t('stage_register', lang)}[/bold yellow]")
        registry = SeedRegistry(registry_dir="./seeds", lang=lang)
        
        table = Table(title="Results", show_header=True, header_style="bold magenta")
        table.add_column("Layer", justify="center")
        table.add_column("Neurons", justify="center")
        table.add_column("Indices", style="cyan")
        table.add_column("Scale", justify="center")
        
        proof_base = f"run_{int(time.time())}"
        for layer_idx, indices in master_neurons.items():
            registry.register_proof(
                proof_name=f"{proof_base}_layer{layer_idx}",
                model_type=orchestrator.model.config.model_type,
                layer_idx=layer_idx,
                master_indices=indices,
                scale=config["scale"],
                model_hidden_size=getattr(orchestrator.model.config, "hidden_size", None),
                model_name=config["model_id"],
                scaled_model_path="./scaled_model"
            )
            orchestrator.apply_scaling(layer_idx, indices, config["scale"])
            
            idx_str = ", ".join(map(str, indices[:5])) + ("..." if len(indices) > 5 else "")
            table.add_row(str(layer_idx), str(len(indices)), idx_str, f"x{config['scale']}")
        
        console.print(table)
        
        # Сохранение
        save_path = "./scaled_model"
        console.print(f"\n[bold yellow]{get_t('stage_save', lang, path=save_path)}[/bold yellow]")
        os.makedirs(save_path, exist_ok=True)
        orchestrator.model.save_pretrained(save_path)
        orchestrator.tokenizer.save_pretrained(save_path)
        console.print("[green]✅ Saved[/green]")
        
        # Генерация
        console.print(f"\n[bold yellow]{get_t('stage_test', lang)}[/bold yellow]")
        orchestrator.model.eval()
        inputs = orchestrator.tokenizer(config["test_prompt"], return_tensors="pt").to(orchestrator.model.device)
        
        with torch.no_grad():
            outputs = orchestrator.model.generate(
                **inputs, max_new_tokens=60, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=orchestrator.tokenizer.eos_token_id
            )
        
        response = orchestrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        console.print(Panel(f"[white]{response}[/white]", title="Model response", border_style="green"))
        
        console.print(Panel(
            get_t('success_text', lang, path=save_path, count=len(master_neurons)),
            title=f"[bold green]{get_t('success_title', lang)}[/bold green]",
            border_style="green"
        ))
        
        return save_path
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]{get_t('error_interrupted', lang)}[/yellow]")
        return None
    except Exception as e:
        console.print(f"\n[red]{get_t('error_critical', lang, error=str(e))}[/red]")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ С ЛУПОМ
# ============================================================

def main():
    try:
        # 1. Выбор языка
        lang = choose_language()
        
        # 2. Онбординг
        show_onboarding(lang)
        
        # 3. Основной луп
        while True:
            # Сбор конфигурации
            model_id = choose_model(lang)
            if not model_id:
                if Confirm.ask("Try again?", default=True):
                    continue
                break
            
            dataset_path = choose_dataset(lang)
            layers = choose_depth(lang, model_id)
            scale, top_k = choose_intensity(lang)
            test_prompt = choose_prompt(lang)
            
            config = {
                "model_id": model_id,
                "dataset_path": dataset_path,
                "layers": layers,
                "scale": scale,
                "top_k": top_k,
                "batch_size": 2 if "0.5B" in model_id or "1.5B" in model_id else 1,
                "test_prompt": test_prompt
            }
            
            # Запуск эксперимента
            save_path = run_experiment(config, lang)
            
            if save_path:
                console.print("\n[bold green]✅ ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН![/bold green]" if lang == "ru" else "\n[bold green]✅ EXPERIMENT SUCCESSFULLY COMPLETED![/bold green]")
                
                if lang == "ru":
                    console.print(f"• Семена сохранены в папке: ./seeds/")
                    console.print(f"• Модель сохранена в: {save_path}")
                    console.print("\n[bold cyan]Для интерактивного тестирования семян запустите:[/bold cyan]")
                    console.print("  python test_seeds.py")
                    console.print("\n[bold]Спасибо за использование MeaningSeed![/bold]")
                    return
                else:
                    console.print(f"• Seeds saved to: ./seeds/")
                    console.print(f"• Model saved to: {save_path}")
                    console.print("\n[bold cyan]To interactively test the seeds, run:[/bold cyan]")
                    console.print("  python test_seeds.py")
                    console.print("\n[bold]Thank you for using MeaningSeed![/bold]")
                    return
            else:
                console.print(f"\n[red]{get_t('error_critical', lang, error='Experiment failed or was interrupted.')}[/red]")
                    
    except KeyboardInterrupt:
        console.print(f"\n[yellow]{get_t('error_interrupted', lang)}[/yellow]")
    except Exception as e:
        console.print(f"\n[red]{get_t('error_critical', lang, error=escape(str(e)))}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
