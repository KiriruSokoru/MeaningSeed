# MeaningSeed: Real-World Proof Scaling

Инструмент для поиска, масштабирования и сохранения "мастер-нейронов" (Real-World Proof) в весах больших языковых моделей (LLM).

Изначально протестировано на `distilgpt2`, архитектура успешно масштабирована на **Qwen2.5-0.5B** и готова к расширению на Llama-3, Mistral и другие архитектуры через паттерн Adapter.

## Возможности

- **Model Adapter Pattern:** Единый интерфейс для работы с разными архитектурами MLP (GPT-2, Qwen2/SwiGLU).
- **Activation Extraction:** Автоматический поиск топ-K наиболее активных нейронов на репрезентативном датасете.
- **Weight Scaling:** Математически корректное масштабирование весов `gate`, `up` и `down` проекций без нарушения размерностей.
- **Seed Registry:** Сохранение и валидация "семян" (proofs) с привязкой к `model_type` для безопасности.
- **Rich CLI:** Визуализация процесса с прогресс-барами и таблицами.

## Установка

```bash
git clone https://github.com/YOUR_USERNAME/MeaningSeed.git
cd MeaningSeed
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt
Использование
Запуск поиска и масштабирования мастер-нейронов:
python main.py --layers 10 12 14 --top_k 10 --scale 1.5
Основные аргументы:

    --model: Название модели Hugging Face (по умолчанию: Qwen/Qwen2.5-0.5B-Instruct)
    --layers: Индексы слоев для анализа (например, 10 12 14)
    --top_k: Количество мастер-нейронов для сохранения (по умолчанию: 10)
    --scale: Коэффициент усиления (>1.0) или подавления (<1.0)
    --save_path: Путь для сохранения модифицированной модели
    --test_prompt: Промпт для тестовой генерации

Структура проекта

    meaning_seed/ — ядро библиотеки (адаптеры, оркестратор, экстрактор, реестр)
    main.py — CLI-скрипт для запуска полного пайплайна
    seeds/ — сохранённые JSON-доказательства (Real-World Proofs)

Добавление поддержки новой архитектуры
Создайте новый класс-наследник BaseModelAdapter в meaning_seed/model_adapter.py, реализуйте 6 абстрактных методов и добавьте условие в функцию get_model_adapter().
Лицензия
MIT
