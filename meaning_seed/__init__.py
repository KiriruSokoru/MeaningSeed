"""
MeaningSeed: Real-World Proof для языковых моделей.
Масштабируемая архитектура с поддержкой GPT-2, Qwen2 и других.
"""

from .orchestrator import Orchestrator
from .extractor import ActivationExtractor
from .registry import SeedRegistry
from .model_adapter import (
    get_model_adapter, 
    BaseModelAdapter, 
    GPT2Adapter, 
    Qwen2Adapter
)

__version__ = "0.2.0"
__all__ = [
    "Orchestrator",
    "ActivationExtractor",
    "SeedRegistry",
    "get_model_adapter",
    "BaseModelAdapter",
    "GPT2Adapter",
    "Qwen2Adapter"
]
