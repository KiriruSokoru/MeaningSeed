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
        "test_title": "🔑 Интерактивное тестирование семян",
        "test_no_seeds": "❌ В папке ./seeds/ нет JSON файлов. Сначала запустите main.py",
        "test_available": "Доступные семена:",
        "test_select": "Выберите номер семени",
        "test_seed_info": "Тип модели в семени: {model_type}",
        "test_model_prompt": "Введите ID модели для загрузки",
        "test_loading": "⚡ Загрузка модели и применение семени...",
        "test_applied": "✅ Семя применено! (Слой {layer}, {count} нейронов, масштаб x{scale})",
        "test_chat": "💬 Интерактивный чат. Введите 'выход' или 'exit' для завершения.",
        "test_you": "Вы",
        "test_model": "Модель",
        "test_ended": "Сеанс завершен."
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
        "test_title": "🔑 Interactive Seed Testing",
        "test_no_seeds": "❌ No JSON files in ./seeds/. Please run main.py first.",
        "test_available": "Available seeds:",
        "test_select": "Select seed number",
        "test_seed_info": "Model type in seed: {model_type}",
        "test_model_prompt": "Enter model ID to load",
        "test_loading": "⚡ Loading model and applying seed...",
        "test_applied": "✅ Seed applied! (Layer {layer}, {count} neurons, scale x{scale})",
        "test_chat": "💬 Interactive chat. Type 'exit' or 'выход' to quit.",
        "test_you": "You",
        "test_model": "Model",
        "test_ended": "Session ended."

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
