# meaning_seed/i18n.py

I18N = {
    "ru": {
        # Orchestrator
        "loading_tokenizer_model": "🔄 Загрузка токенизатора и модели: {model_name}...",
        "model_loaded_arch": "✅ Модель загружена. Архитектура: {model_type}",
        "using_adapter": "🔧 Используется адаптер: {adapter_name}",
        "total_layers": "📊 Всего слоёв: {num_layers}",
        "scaling_applied": "⚡ Масштабирование x{scale} применено к слою {layer_idx}, нейроны: {indices}",
        "seed_saved": "💾 Семя сохранено в {filepath}",
        "seed_applied": "✅ Семя из {filepath} успешно применено.",
        "arch_mismatch": "❌ Несоответствие архитектур! Семя создано для: {seed_type}, текущая модель: {current_type}",
        
        # Extractor
        "collecting_activations": "🧠 Сбор активаций MLP",
        "batch": "batch",
        "master_neurons_found": "🔍 Слой {layer_idx}: Топ-{top_k} мастер-нейронов найдены. Макс. активация: {max_act:.4f}",
        
        # Registry
        "proof_registered": "✅ Proof '{proof_name}' зарегистрирован и сохранен в {filepath}",
        "error_reading_file": "⚠️ Ошибка чтения файла: {filename}",
        
        # Model Adapter
        "unsupported_arch": "Неподдерживаемая архитектура модели: {model_type}. Поддерживаются: gpt2, qwen2",
        "model_options": "  [1] [bold green]Qwen2.5-0.5B[/bold green] (рекомендуется для CPU, ~1GB RAM)\n  [2] [bold yellow]Qwen2.5-1.5B[/bold yellow] (требуется ~3GB свободной RAM)\n  [3] [bold yellow]Qwen2.5-3B[/bold yellow] (требуется ~6GB свободной RAM)\n  [4] [bold red]Своя модель с HuggingFace[/bold red] (укажите ID)",
        "dataset_options": "  [1] Быстрый демо-набор (50 фраз, ~1 минута)\n  [2] [bold]Реальный набор Alpaca[/bold] (200 инструкций, ~15-20 минут)\n  [3] Свой JSON файл",
        "depth_options": "  [1] Быстрый тест (один средний слой)\n  [2] [bold]Полный анализ[/bold] (все слои модели — даст полную карту)\n  [3] Выбранные слои (введите через пробел, например: 10 12 14)",
        "intensity_options": "  [1] [bold green]Сбалансированная[/bold green] (x1.5, топ-10 нейронов) — рекомендуется\n  [2] [bold yellow]Агрессивная[/bold yellow] (x2.0, топ-20 нейронов) — возможны галлюцинации\n  [3] [bold cyan]Подавление[/bold cyan] (x0.5, топ-10) — делает ответы шаблоннее",
        "menu_options": "  [1] [bold green]🧪 Протестировать созданные семена[/bold green] (интерактивный чат)\n  [2] [bold cyan]🔄 Повторить эксперимент[/bold cyan] (с другими параметрами)\n  [3] [bold red]🚪 Завершить работу[/bold red]",

        #Seeds
        "seed_path_prompt": "Введите путь к JSON-сиду (или оставьте пустым для сканирования {default})",
        "no_seeds_found": "В указанной директории не найдено JSON-сидов.",
        "available_seeds": "Доступные сиды:",
        "seed_name": "Имя сида",
        "target_model": "Целевая модель",
        "select_seed": "Выберите номер сида",
        "invalid_choice": "Неверный выбор.",
        "seed_loaded": "Сид загружен:",
        "scaling_factor": "Коэффициент масштабирования",
        "model_unknown_warning": "⚠️ В этом сиду не указано имя модели (старый формат). Укажите его вручную.",
        "enter_model_id_or_path": "Введите ID модели (HuggingFace) или путь к локальной папке",
        "scaled_model_found": "Найдена предварительно масштабированная модель:",
        "load_scaled_directly": "Загрузить её напрямую? (Рекомендуется для скорости)",
        "will_load_base": "Будет загружена базовая модель с применением сида на лету.",
        "no_scaled_model": "Предварительно масштабированная модель не найдена. Будет использована базовая модель.",
        "override_model_prompt": "Переопределить ID/путь модели? (Оставьте пустым для: {target})",
        "strict_validation_warning": "Внимание: При ручном переопределении будет выполнена строгая проверка размерностей.",
        "loading_model": "Загрузка модели:",
        "model_load_error": "Не удалось загрузить модель: {error}",
        "checking_compatibility": "Проверка совместимости сида...",
        "compatibility_error": "Ошибка совместимости:",
        "application_aborted": "Применение прервано для предотвращения краша тензоров.",
        "compatibility_confirmed": "Совместимость подтверждена. Применение сида на лету...",
        "seed_apply_error": "Ошибка применения сида: {error}",
        "scaled_model_loaded": "Загружена предварительно масштабированная модель. Сид уже применен.",
        "interactive_generation": "Интерактивный цикл генерации",
        "commands_info": "Команды: 'exit' (выход), 'clear' (очистить историю)",
        "you_prompt": "Вы:",
        "goodbye": "Завершение работы. До встречи!",
        "history_cleared": "История диалога очищена.",
        "model_prompt": "Модель:",
        "generating": "генерация...",
        "generation_error": "Ошибка генерации: {error}",
        "interrupted": "Прервано пользователем. Завершение.",
        "critical_error": "Критическая ошибка: {error}",
        "seed_read_error": "Ошибка чтения сида: {error}",
    },
    "en": {
        # Orchestrator
        "loading_tokenizer_model": "🔄 Loading tokenizer and model: {model_name}...",
        "model_loaded_arch": "✅ Model loaded. Architecture: {model_type}",
        "using_adapter": "🔧 Using adapter: {adapter_name}",
        "total_layers": "📊 Total layers: {num_layers}",
        "scaling_applied": "⚡ Scaling x{scale} applied to layer {layer_idx}, neurons: {indices}",
        "seed_saved": "💾 Seed saved to {filepath}",
        "seed_applied": "✅ Seed from {filepath} successfully applied.",
        "arch_mismatch": "❌ Architecture mismatch! Seed created for: {seed_type}, current model: {current_type}",
        
        # Extractor
        "collecting_activations": "🧠 Collecting MLP activations",
        "batch": "batch",
        "master_neurons_found": "🔍 Layer {layer_idx}: Top-{top_k} master neurons found. Max activation: {max_act:.4f}",
        
        # Registry
        "proof_registered": "✅ Proof '{proof_name}' registered and saved to {filepath}",
        "error_reading_file": "⚠️ Error reading file: {filename}",
        
        # Model Adapter
        "unsupported_arch": "Unsupported model architecture: {model_type}. Supported: gpt2, qwen2",
        "model_options": "  [1] [bold green]Qwen2.5-0.5B[/bold green] (recommended for CPU, ~1GB RAM)\n  [2] [bold yellow]Qwen2.5-1.5B[/bold yellow] (requires ~3GB free RAM)\n  [3] [bold yellow]Qwen2.5-3B[/bold yellow] (requires ~6GB free RAM)\n  [4] [bold red]Custom model from HuggingFace[/bold red] (specify ID)",
        "dataset_options": "  [1] Quick demo set (50 phrases, ~1 minute)\n  [2] [bold]Real Alpaca dataset[/bold] (200 instructions, ~15-20 minutes)\n  [3] Custom JSON file",
        "depth_options": "  [1] Quick test (one middle layer)\n  [2] [bold]Full analysis[/bold] (all model layers — gives complete map)\n  [3] Custom layers (enter space-separated, e.g.: 10 12 14)",
        "intensity_options": "  [1] [bold green]Balanced[/bold green] (x1.5, top-10 neurons) — recommended\n  [2] [bold yellow]Aggressive[/bold yellow] (x2.0, top-20 neurons) — may cause hallucinations\n  [3] [bold cyan]Suppression[/bold cyan] (x0.5, top-10) — makes responses more template-like",
        "menu_options": "  [1] [bold green]🧪 Test created seeds[/bold green] (interactive chat)\n  [2] [bold cyan]🔄 Repeat experiment[/bold cyan] (with different parameters)\n  [3] [bold red]🚪 Exit[/bold red]",

        #Seeds
        "seed_path_prompt": "Enter path to JSON seed (or leave empty to scan {default})",
        "no_seeds_found": "No JSON seeds found in the specified directory.",
        "available_seeds": "Available seeds:",
        "seed_name": "Seed name",
        "target_model": "Target model",
        "select_seed": "Select seed number",
        "invalid_choice": "Invalid choice.",
        "seed_loaded": "Seed loaded:",
        "scaling_factor": "Scaling factor",
        "model_unknown_warning": "⚠️ This seed does not contain model name (old format). Please specify it manually.",
        "enter_model_id_or_path": "Enter model ID (HuggingFace) or path to local folder",
        "scaled_model_found": "Found pre-scaled model:",
        "load_scaled_directly": "Load it directly? (Recommended for speed)",
        "will_load_base": "Base model will be loaded with seed applied on the fly.",
        "no_scaled_model": "Pre-scaled model not found. Base model will be used.",
        "override_model_prompt": "Override model ID/path? (Leave empty for: {target})",
        "strict_validation_warning": "Warning: Strict dimension validation will be performed on manual override.",
        "loading_model": "Loading model:",
        "model_load_error": "Failed to load model: {error}",
        "checking_compatibility": "Checking seed compatibility...",
        "compatibility_error": "Compatibility error:",
        "application_aborted": "Application aborted to prevent tensor crash.",
        "compatibility_confirmed": "Compatibility confirmed. Applying seed on the fly...",
        "seed_apply_error": "Seed application error: {error}",
        "scaled_model_loaded": "Pre-scaled model loaded. Seed is already applied.",
        "interactive_generation": "Interactive generation loop",
        "commands_info": "Commands: 'exit' (quit), 'clear' (clear history)",
        "you_prompt": "You:",
        "goodbye": "Exiting. Goodbye!",
        "history_cleared": "Dialog history cleared.",
        "model_prompt": "Model:",
        "generating": "generating...",
        "generation_error": "Generation error: {error}",
        "interrupted": "Interrupted by user. Exiting.",
        "critical_error": "Critical error: {error}",
        "seed_read_error": "Seed read error: {error}",

    }
}

def get_t(key, lang="ru", **kwargs):
    """Get translated string"""
    text = I18N.get(lang, I18N["ru"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text
