"""
MeaningSeed: Topological orchestration of LLM tasks via Ricci curvature.
"""

__version__ = "0.1.0"
__author__ = "Kirill Sokol (KiriruSokoru)"

from .orchestrator import Orchestrator
from .extractor import MeaningExtractor
from .registry import SeedRegistry

__all__ = ["Orchestrator", "MeaningExtractor", "SeedRegistry"]
