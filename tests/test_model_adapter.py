"""
Тесты для модуля model_adapter.

Проверяет фабричную функцию get_model_adapter() и корректность
возвращаемых адаптеров для различных архитектур моделей.
"""

import pytest
from unittest.mock import MagicMock

from meaning_seed.model_adapter import (
    get_model_adapter,
    GPT2Adapter,
    Qwen2Adapter,
    BaseModelAdapter,
)


class TestGetModelAdapter:
    """Тесты для фабричной функции get_model_adapter()."""

    def test_gpt2_adapter_returns_correct_type(self) -> None:
        """Проверяет, что для GPT2Config возвращается GPT2Adapter."""
        # Создаём мок модели с конфигурацией GPT-2
        mock_model = MagicMock()
        mock_model.config.model_type = "gpt2"
        
        adapter = get_model_adapter(mock_model, lang="ru")
        
        assert isinstance(adapter, GPT2Adapter)
        assert isinstance(adapter, BaseModelAdapter)
        assert adapter.lang == "ru"

    def test_qwen2_adapter_returns_correct_type(self) -> None:
        """Проверяет, что для Qwen2Config возвращается Qwen2Adapter."""
        # Создаём мок модели с конфигурацией Qwen2
        mock_model = MagicMock()
        mock_model.config.model_type = "qwen2"
        
        adapter = get_model_adapter(mock_model, lang="en")
        
        assert isinstance(adapter, Qwen2Adapter)
        assert isinstance(adapter, BaseModelAdapter)
        assert adapter.lang == "en"

    def test_qwen2_adapter_with_qwen2_5_type(self) -> None:
        """Проверяет, что 'qwen2_5' тоже возвращает Qwen2Adapter."""
        mock_model = MagicMock()
        mock_model.config.model_type = "qwen2_5"
        
        adapter = get_model_adapter(mock_model, lang="ru")
        
        assert isinstance(adapter, Qwen2Adapter)

    def test_unsupported_arch_raises_error(self) -> None:
        """Проверяет ValueError для неизвестной архитектуры."""
        mock_model = MagicMock()
        mock_model.config.model_type = "unknown_architecture"
        
        with pytest.raises(ValueError, match="Неподдерживаемая архитектура модели"):
            get_model_adapter(mock_model, lang="ru")

    def test_case_insensitive_model_type(self) -> None:
        """Проверяет, что проверка model_type нечувствительна к регистру."""
        mock_model = MagicMock()
        mock_model.config.model_type = "GPT2"  # Uppercase
        
        adapter = get_model_adapter(mock_model, lang="ru")
        
        assert isinstance(adapter, GPT2Adapter)

    def test_missing_model_type_attribute(self) -> None:
        """Проверяет поведение при отсутствии атрибута model_type."""
        mock_model = MagicMock(spec=[])  # Пустой spec — нет атрибутов
        
        with pytest.raises(ValueError, match="Неподдерживаемая архитектура модели"):
            get_model_adapter(mock_model, lang="ru")


class TestAdapterInterface:
    """Тесты для проверки интерфейса адаптеров."""

    def test_gpt2_adapter_has_required_methods(self) -> None:
        """Проверяет, что GPT2Adapter реализует все абстрактные методы."""
        adapter = GPT2Adapter(lang="ru")
        
        assert hasattr(adapter, 'get_num_layers')
        assert hasattr(adapter, 'get_mlp_weights')
        assert hasattr(adapter, 'register_intermediate_hook')
        assert hasattr(adapter, 'scale_master_neurons')
        assert callable(adapter.get_num_layers)

    def test_qwen2_adapter_has_required_methods(self) -> None:
        """Проверяет, что Qwen2Adapter реализует все абстрактные методы."""
        adapter = Qwen2Adapter(lang="en")
        
        assert hasattr(adapter, 'get_num_layers')
        assert hasattr(adapter, 'get_mlp_weights')
        assert hasattr(adapter, 'register_intermediate_hook')
        assert hasattr(adapter, 'scale_master_neurons')
        assert callable(adapter.get_num_layers)
