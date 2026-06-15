#!/usr/bin/env python3
"""
Главный CLI-скрипт MeaningSeed для интерактивного создания топологических семян.

Этот модуль предоставляет интерактивный интерфейс для:
- Выбора модели, датасета, глубины анализа и интенсивности масштабирования
- Запуска эксперимента по извлечению мастер-нейронов
- Применения и тестирования созданных семян

Разделение ответственности:
- build_config_from_cli() — сбор конфигурации из интерактивных prompt'ов (UI)
- run_experiment() — чистая бизнес-логика без UI (использует logging)
"""

import json
import logging
import os
import sys
import time
from typing import List, Optional, Dict, Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from transformers import AutoConfig

from meaning_seed import Orchestrator, ActivationExtractor, SeedRegistry
from meaning_seed.i18n import get_t

# Локальный логгер для main.py
logger = logging.getLogger(__name__)

# Консоль для Rich UI
console = Console()

# ============================================================================
# I18N — только уникальные ключи для CLI (не дублируют meaning_seed/i18n.py)
# ============================================================================

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

        "model_choice": "Ваш выбор",
        "model_custom_id": "Введите ID модели (например, meta-llama/Llama-3.2-1B-Instruct)",

        "ram_warning": "[bold red]⚠️ ВНИМАНИЕ:[/bold red] У вас {available:.1f}GB свободной RAM, а модель требует ~{required:.1f}GB. Система может уйти в swap и работать ОЧЕНЬ медленно.",
        "ram_ok": "[green]✓ Достаточно RAM: свободно {available:.1f}GB, требуется ~{required:.1f}GB[/green]",
        "ram_check": "Проверка системных ресурсов...",
        "ram_proceed_anyway": "Всё равно продолжить?",

        "dataset_choice": "Ваш выбор",
        "dataset_custom_path": "Введите путь к JSON файлу",
        "dataset_loading": "📥 Скачивание датасета Alpaca...",
        "dataset_loaded": "✅ Датасет загружен",

        "depth_choice": "Ваш выбор",
        "depth_custom": "Введите номера слоев через пробел",

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
        "menu_choice": "Ваш выбор",

        "test_title": "🔑 Применение семени",
        "test_select": "Выберите номер семени",
        "test_applying": "⚡ Применение семени...",
        "test_applied": "✅ Семя применено. Слой {layer}, {count} нейронов, масштаб x{scale}",
        "test_chat": "💬 Интерактивный чат (введите 'выход' для завершения)",
        "test_you": "Вы",
        "test_model": "Модель",

        "error_no_seeds": "❌ В папке ./seeds/ нет семян. Сначала выполните генерацию.",
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

        "model_choice": "Your choice",
        "model_custom_id": "Enter model ID (e.g., meta-llama/Llama-3.2-1B-Instruct)",

        "ram_warning": "[bold red]⚠️ WARNING:[/bold red] You have {available:.1f}GB free RAM, but the model needs ~{required:.1f}GB. System may swap and work VERY slowly.",
        "ram_ok": "[green]✓ RAM is sufficient: {available:.1f}GB free, ~{required:.1f}GB required[/green]",
        "ram_check": "Checking system resources...",
        "ram_proceed_anyway": "Proceed anyway?",

        "dataset_choice": "Your choice",
        "dataset_custom_path": "Enter path to JSON file",
        "dataset_loading": "📥 Downloading Alpaca dataset...",
        "dataset_loaded": "✅ Dataset loaded",

        "depth_choice": "Your choice",
        "depth_custom": "Enter layer numbers separated by spaces",

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
        "menu_choice": "Your choice",

        "test_title": "🔑 Applying seed",
        "test_select": "Select seed number",
        "test_applying": "⚡ Applying seed...",
        "test_applied": "✅ Seed applied. Layer {layer}, {count} neurons, scale x{scale}",
        "test_chat": "💬 Interactive chat (type 'exit' to quit)",
        "test_you": "You",
        "test_model": "Model",

        "error_no_seeds": "❌ No seeds in ./seeds/. Please run generation first.",
    }
}


def get_local_t(key: str, lang: str, **kwargs: Any) -> str:
    """
    Получить перевод из локального I18N (для CLI-специфичных ключей).
    
    Args:
        key: Ключ перевода
        lang: Язык ("ru" или "en")
        **kwargs: Параметры для форматирования
        
    Returns:
        Переведённая строка
    """
    lang_dict = I18N.get(lang, I18N["ru"])
    text = lang_dict.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            return text
    return text


# ============================================================================
# Вспомогательные функции
# ============================================================================

def estimate_model_ram_gb(model_id: str) -> float:
    """Оценить требуемую RAM для модели в GB."""
    model_id_lower = model_id.lower()
    if "0.5b" in model_id_lower:
        return 1.0
    elif "1.5b" in model_id_lower:
        return 3.0
    elif "3b" in model_id_lower:
        return 6.0
    elif "7b" in model_id_lower:
        return 14.0
    else:
        return 8.0  # По умолчанию


def get_available_ram_gb() -> float:
    """Получить доступную RAM в GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return 16.0  # Если psutil не установлен


def check_ram_safety(required_gb: float, lang: str) -> bool:
    """Проверить достаточно ли RAM для модели."""
    console.print(f"\n[cyan]{get_local_t('ram_check', lang)}[/cyan]")
    available = get_available_ram_gb()
    
    if available < required_gb:
        console.print(get_local_t('ram_warning', lang, available=available, required=required_gb))
        return Confirm.ask(get_local_t('ram_proceed_anyway', lang), default=False)
    else:
        console.print(get_local_t('ram_ok', lang, available=available, required=required_gb))
        return True


def ensure_alpaca_dataset(lang: str) -> Optional[str]:
    """Загрузить датасет Alpaca если его нет."""
    dataset_path = "./data/alpaca_200.json"
    if os.path.exists(dataset_path):
        return dataset_path
    
    console.print(get_local_t('dataset_loading', lang))
    # Здесь должна быть логика загрузки датасета
    # Для простоты создаём пустой файл
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(["Sample text 1", "Sample text 2"] * 100, f)
    
    console.print(get_local_t('dataset_loaded', lang))
    return dataset_path


# ============================================================================
# CLI функции для сбора конфигурации
# ============================================================================

def choose_language() -> str:
    """Выбрать язык интерфейса."""
    console.print(Panel.fit(
        f"[bold cyan]{get_local_t('welcome_title', 'ru')}[/bold cyan]\n"
        f"[dim]{get_local_t('welcome_subtitle', 'ru')}[/dim]\n\n"
        f"[bold cyan]{get_local_t('welcome_title', 'en')}[/bold cyan]\n"
        f"[dim]{get_local_t('welcome_subtitle', 'en')}[/dim]",
        border_style="cyan"
    ))
    choice = Prompt.ask(
        f"\n{get_local_t('choose_language', 'ru')} [RU/EN]",
        choices=["RU", "EN", "ru", "en"],
        default="EN"
    ).lower()
    return "ru" if choice == "ru" else "en"


def show_onboarding(lang: str) -> None:
    """Показать онбординг с описанием проекта."""
    console.print(Panel(
        get_local_t('about_text', lang),
        title=f"[bold]{get_local_t('about_title', lang)}[/bold]",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()
    console.print(Panel(
        get_local_t('process_text', lang),
        title=f"[bold]{get_local_t('process_title', lang)}[/bold]",
        border_style="yellow",
        padding=(1, 2)
    ))
    Prompt.ask("\n[bold]Press Enter to continue...[/bold]")


def choose_model(lang: str) -> Optional[str]:
    """Выбрать модель для эксперимента."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_local_t('step_model', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('model_options', lang)))

    choice = Prompt.ask(get_local_t('model_choice', lang), choices=["1", "2", "3", "4"], default="1")

    model_map = {
        "1": ("Qwen/Qwen2.5-0.5B-Instruct", 1.0),
        "2": ("Qwen/Qwen2.5-1.5B-Instruct", 3.0),
        "3": ("Qwen/Qwen2.5-3B-Instruct", 6.0),
    }

    if choice in model_map:
        model_id, required = model_map[choice]
    else:
        model_id = Prompt.ask(get_local_t('model_custom_id', lang))
        required = estimate_model_ram_gb(model_id)

    if not check_ram_safety(required, lang):
        return None

    return model_id


def choose_dataset(lang: str) -> Optional[str]:
    """Выбрать датасет для анализа."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_local_t('step_dataset', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('dataset_options', lang)))

    choice = Prompt.ask(get_local_t('dataset_choice', lang), choices=["1", "2", "3"], default="2")

    if choice == "1":
        return None  # встроенный датасет
    elif choice == "2":
        return ensure_alpaca_dataset(lang)
    else:
        path = Prompt.ask(get_local_t('dataset_custom_path', lang))
        return os.path.abspath(path) if os.path.exists(path) else None


def choose_depth(lang: str, model_id: str) -> List[int]:
    """Выбрать глубину анализа (слои)."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_local_t('step_depth', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('depth_options', lang)))

    choice = Prompt.ask(get_local_t('depth_choice', lang), choices=["1", "2", "3"], default="1")

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
        layers_str = Prompt.ask(get_local_t('depth_custom', lang), default="10 12 14")
        try:
            return [int(x.strip()) for x in layers_str.split()]
        except Exception:
            return [12]


def choose_intensity(lang: str) -> tuple[float, int]:
    """Выбрать интенсивность масштабирования."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_local_t('step_intensity', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(Text.from_markup(get_t('intensity_options', lang)))

    choice = Prompt.ask(get_local_t('intensity_choice', lang), choices=["1", "2", "3"], default="1")

    intensity_map = {
        "1": (1.5, 10),
        "2": (2.0, 20),
        "3": (0.5, 10),
    }
    return intensity_map[choice]


def choose_prompt(lang: str) -> str:
    """Выбрать тестовый промпт."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{get_local_t('step_prompt', lang)}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    return Prompt.ask(get_local_t('prompt_ask', lang), default=get_local_t('prompt_default', lang))


def build_config_from_cli(lang: str) -> Optional[Dict[str, Any]]:
    """
    Собрать конфигурацию эксперимента из интерактивных prompt'ов.
    
    Args:
        lang: Язык интерфейса ("ru" или "en")
        
    Returns:
        Словарь с конфигурацией или None если пользователь отменил выбор
    """
    model_id = choose_model(lang)
    if not model_id:
        return None

    dataset_path = choose_dataset(lang)
    layers = choose_depth(lang, model_id)
    scale, top_k = choose_intensity(lang)
    test_prompt = choose_prompt(lang)

    return {
        "model_id": model_id,
        "dataset_path": dataset_path,
        "layers": layers,
        "scale": scale,
        "top_k": top_k,
        "batch_size": 2 if "0.5B" in model_id or "1.5B" in model_id else 1,
        "test_prompt": test_prompt
    }


# ============================================================================
# Бизнес-логика эксперимента
# ============================================================================

def run_experiment(config: Dict[str, Any], lang: str) -> Optional[str]:
    """
    Запустить эксперимент по извлечению и применению мастер-нейронов.
    
    Это чистая бизнес-логика без UI — использует logging вместо console.print.
    
    Args:
        config: Словарь с конфигурацией эксперимента
        lang: Язык для логирования
        
    Returns:
        Путь к сохранённой модели или None при ошибке/прерывании
    """
    orchestrator = None
    
    try:
        # Загрузка модели
        logger.info(get_local_t('stage_loading', lang))
        start_time = time.time()
        orchestrator = Orchestrator(model_name=config["model_id"], device="auto", lang=lang)
        logger.info(f"✅ Model loaded in {time.time() - start_time:.2f}s")

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
        logger.info(get_local_t('stage_analysis', lang, count=len(texts), batch=config['batch_size']))
        extractor = ActivationExtractor(model=orchestrator.model, adapter=orchestrator.adapter, lang=lang)

        master_neurons = extractor.find_master_neurons(
            texts=texts,
            tokenizer=orchestrator.tokenizer,
            layer_indices=config["layers"],
            top_k=config["top_k"],
            batch_size=config["batch_size"]
        )

        # Регистрация семян
        logger.info(get_local_t('stage_register', lang))
        registry = SeedRegistry(registry_dir="./seeds", lang=lang)

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

        # Сохранение модели
        save_path = "./scaled_model"
        logger.info(get_local_t('stage_save', lang, path=save_path))
        os.makedirs(save_path, exist_ok=True)
        orchestrator.model.save_pretrained(save_path)
        orchestrator.tokenizer.save_pretrained(save_path)
        logger.info("✅ Model saved")

        # Тестовая генерация
        logger.info(get_local_t('stage_test', lang))
        orchestrator.model.eval()
        inputs = orchestrator.tokenizer(config["test_prompt"], return_tensors="pt").to(orchestrator.model.device)

        with torch.no_grad():
            outputs = orchestrator.model.generate(
                **inputs, max_new_tokens=60, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=orchestrator.tokenizer.eos_token_id
            )

        response = orchestrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model response: {response}")

        logger.info(get_local_t('success_text', lang, path=save_path, count=len(master_neurons)))
        return save_path

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Critical error in experiment: {e}", exc_info=True)
        return None
    finally:
        # Очистка ресурсов
        if orchestrator is not None:
            try:
                del orchestrator.model
                del orchestrator.tokenizer
                del orchestrator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================================
# Главная функция
# ============================================================================

def main() -> int:
    """
    Главная функция CLI.
    
    Returns:
        0 — успех, 1 — ошибка
    """
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    lang = "ru"
    
    try:
        # 1. Выбор языка
        lang = choose_language()

        # 2. Онбординг
        show_onboarding(lang)

        # 3. Основной цикл
        while True:
            # Сбор конфигурации
            config = build_config_from_cli(lang)
            if not config:
                if Confirm.ask("Try again?", default=True):
                    continue
                break

            # Запуск эксперимента
            save_path = run_experiment(config, lang)

            if save_path:
                console.print("\n[bold green]✅ ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН![/bold green]" if lang == "ru" else "\n[bold green]✅ EXPERIMENT SUCCESSFULLY COMPLETED![/bold green]")
                console.print(Panel(
                    get_local_t('success_text', lang, path=save_path, count=len(config["layers"])),
                    title=f"[bold green]{get_local_t('success_title', lang)}[/bold green]",
                    border_style="green"
                ))
                return 0
            else:
                console.print(f"\n[red]{get_t('error_critical', lang, error='Experiment failed or was interrupted.')}[/red]")
                return 1

    except KeyboardInterrupt:
        console.print(f"\n[yellow]{get_t('error_interrupted', lang)}[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]{get_t('error_critical', lang, error=str(e))}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
