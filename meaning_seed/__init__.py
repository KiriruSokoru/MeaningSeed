"""
MeaningSeed — библиотека для хирургического вмешательства в веса LLM.

Пакет предоставляет инструменты для извлечения, сохранения и применения 
топологических семян (мастер-нейронов) к языковым моделям без необходимости 
их переобучения (fine-tuning).

Основные компоненты:
- Orchestrator: Управление загрузкой модели, применением семян и генерацией.
- ActivationExtractor: Поиск мастер-нейронов через анализ активаций MLP слоёв.
- model_adapter: Адаптеры для унифицированной работы с разными архитектурами (GPT-2, Qwen2).
- registry: Управление, валидация и кэширование файлов семян (proofs).
- i18n: Система локализации интерфейсных сообщений.
"""

import logging

# Стандартный паттерн для библиотек: добавляем NullHandler, 
# чтобы библиотека не перехватывала и не засоряла вывод логов пользователя,
# если он явно не настроил логирование для "meaning_seed".
logging.getLogger("meaning_seed").addHandler(logging.NullHandler())

__version__ = "0.2.0"

from .orchestrator import Orchestrator
from .extractor import ActivationExtractor
from .model_adapter import (
    BaseModelAdapter,
    GPT2Adapter,
    Qwen2Adapter,
    get_model_adapter,
)
from .registry import (
    load_seed,
    validate_seed_compatibility,
    save_seed,
    SeedRegistry,
)
from .i18n import get_t

__all__ = [
    "__version__",
    "Orchestrator",
    "ActivationExtractor",
    "BaseModelAdapter",
    "GPT2Adapter",
    "Qwen2Adapter",
    "get_model_adapter",
    "load_seed",
    "validate_seed_compatibility",
    "save_seed",
    "SeedRegistry",
    "get_t",
]
