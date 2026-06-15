import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable

__all__ = ["BaseModelAdapter", "GPT2Adapter", "Qwen2Adapter", "get_model_adapter"]

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Абстрактный базовый класс для адаптеров архитектур языковых моделей.
    
    Определяет единый интерфейс для взаимодействия с различными архитектурами 
    (например, GPT-2, Qwen2), скрывая специфику именования слоёв и структуры весов.
    Это позволяет оркестратору и экстрактору работать с любой поддерживаемой 
    моделью через единый API.
    """

    def __init__(self, lang: str = "ru") -> None:
        """
        Инициализация адаптера.
        
        Args:
            lang: Язык для сообщений и логирования ("ru" или "en")
        """
        self.lang = lang

    @abstractmethod
    def get_num_layers(self, model: nn.Module) -> int:
        """
        Возвращает общее количество трансформер-слоёв в модели.
        
        Args:
            model: Экземпляр языковой модели
            
        Returns:
            Количество слоёв (int)
        """
        pass

    @abstractmethod
    def get_mlp_weights(self, model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает словарь с тензорами весов MLP указанного слоя.
        
        Args:
            model: Экземпляр языковой модели
            layer_idx: Индекс слоя
            
        Returns:
            Словарь вида {"layer_name": torch.Tensor} с весами MLP
        """
        pass

    @abstractmethod
    def register_intermediate_hook(
        self, model: nn.Module, layer_idx: int, hook_fn: Callable
    ) -> Any:
        """
        Регистрирует forward hook на промежуточные активации MLP слоя.
        
        Args:
            model: Экземпляр языковой модели
            layer_idx: Индекс слоя
            hook_fn: Функция хука, принимающая (module, input, output)
            
        Returns:
            Объект хука (RemovableHandle), который можно удалить вызовом .remove()
        """
        pass

    def scale_master_neurons(
        self, model: nn.Module, layer_idx: int, master_indices: List[int], scale: float
    ) -> None:
        """
        Применяет хирургическое масштабирование весов мастер-нейронов.
        
        Математическое обоснование масштабирования:
        Веса стандартного линейного слоя nn.Linear(in_features, out_features) в PyTorch 
        имеют форму [out_features, in_features].
        
        1. Для слоёв расширения (c_fc, gate_proj, up_proj):
           - Форма тензора весов: [intermediate_size, hidden_size].
           - Каждая СТРОКА (dim=0) соответствует весам одного конкретного нейрона 
             промежуточного слоя. 
           - Масштабирование строки умножает вклад всех входных признаков на этот 
             нейрон, что напрямую и линейно масштабирует его выходную активацию 
             перед функцией активации.
        
        2. Для слоёв сжатия (c_proj, down_proj):
           - Форма тензора весов: [hidden_size, intermediate_size].
           - Каждый СТОЛБЕЦ (dim=1) соответствует весам, с которыми один конкретный 
             нейрон промежуточного слоя влияет на все выходные признаки скрытого 
             состояния (hidden state).
           - Масштабирование столбца усиливает или ослабляет вклад активации этого 
             нейрона в итоговое представление, не меняя активацию других нейронов.
        
        Args:
            model: Экземпляр языковой модели
            layer_idx: Индекс слоя для модификации
            master_indices: Список индексов мастер-нейронов для масштабирования
            scale: Коэффициент масштабирования (например, 1.5 для усиления)
        """
        # Базовая реализация переопределяется в конкретных адаптерах
        raise NotImplementedError("Метод должен быть реализован в подклассе")


class GPT2Adapter(BaseModelAdapter):
    """
    Адаптер для архитектуры GPT-2.
    
    Работает со слоями c_fc (расширение) и c_proj (сжатие).
    """

    def get_num_layers(self, model: nn.Module) -> int:
        return model.config.n_layer

    def get_mlp_weights(self, model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
        mlp = model.transformer.h[layer_idx].mlp
        return {
            "c_fc": mlp.c_fc.weight,
            "c_proj": mlp.c_proj.weight
        }

    def register_intermediate_hook(
        self, model: nn.Module, layer_idx: int, hook_fn: Callable
    ) -> Any:
        mlp = model.transformer.h[layer_idx].mlp
        # Хук регистрируется на c_fc для захвата активаций до функции активации
        return mlp.c_fc.register_forward_hook(hook_fn)

    def scale_master_neurons(
        self, model: nn.Module, layer_idx: int, master_indices: List[int], scale: float
    ) -> None:
        mlp = model.transformer.h[layer_idx].mlp
        
        with torch.no_grad():
            # c_fc: [intermediate_size, hidden_size]. Строки (dim=0) = нейроны.
            mlp.c_fc.weight.data[master_indices, :] *= scale
            
            # c_proj: [hidden_size, intermediate_size]. Столбцы (dim=1) = нейроны.
            mlp.c_proj.weight.data[:, master_indices] *= scale


class Qwen2Adapter(BaseModelAdapter):
    """
    Адаптер для архитектуры Qwen2 (и Qwen2.5).
    
    Работает со слоями gate_proj, up_proj (расширение, архитектура SwiGLU) 
    и down_proj (сжатие).
    """

    def get_num_layers(self, model: nn.Module) -> int:
        return model.config.num_hidden_layers

    def get_mlp_weights(self, model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
        mlp = model.model.layers[layer_idx].mlp
        return {
            "gate_proj": mlp.gate_proj.weight,
            "up_proj": mlp.up_proj.weight,
            "down_proj": mlp.down_proj.weight
        }

    def register_intermediate_hook(
        self, model: nn.Module, layer_idx: int, hook_fn: Callable
    ) -> Any:
        mlp = model.model.layers[layer_idx].mlp
        # В Qwen2 (SwiGLU) хук регистрируется на up_proj, так как его выход 
        # (после активации и поэлементного умножения с gate_proj) формирует 
        # промежуточное представление нейрона.
        return mlp.up_proj.register_forward_hook(hook_fn)

    def scale_master_neurons(
        self, model: nn.Module, layer_idx: int, master_indices: List[int], scale: float
    ) -> None:
        mlp = model.model.layers[layer_idx].mlp
        
        with torch.no_grad():
            # gate_proj: [intermediate_size, hidden_size]. Строки (dim=0) = нейроны.
            mlp.gate_proj.weight.data[master_indices, :] *= scale
            
            # up_proj: [intermediate_size, hidden_size]. Строки (dim=0) = нейроны.
            mlp.up_proj.weight.data[master_indices, :] *= scale
            
            # down_proj: [hidden_size, intermediate_size]. Столбцы (dim=1) = нейроны.
            mlp.down_proj.weight.data[:, master_indices] *= scale


def get_model_adapter(model: nn.Module, lang: str = "ru") -> BaseModelAdapter:
    """
    Фабричная функция для создания адаптера модели на основе её конфигурации.
    
    Автоматически определяет архитектуру модели (по полю model_type в config) 
    и возвращает соответствующий экземпляр адаптера.
    
    Args:
        model: Экземпляр языковой модели (transformers.PreTrainedModel)
        lang: Язык для сообщений адаптера ("ru" или "en")
        
    Returns:
        Экземпляр BaseModelAdapter (GPT2Adapter или Qwen2Adapter)
        
    Raises:
        ValueError: Если архитектура модели не поддерживается
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> adapter = get_model_adapter(model, lang="ru")
        >>> print(type(adapter))
        <class 'meaning_seed.model_adapter.Qwen2Adapter'>
    """
    model_type = getattr(model.config, "model_type", "").lower()
    
    if "gpt2" in model_type:
        return GPT2Adapter(lang=lang)
    elif "qwen2" in model_type:
        return Qwen2Adapter(lang=lang)
    else:
        raise ValueError(
            f"Неподдерживаемая архитектура модели: {model_type}. "
            f"Поддерживаются: gpt2, qwen2"
        )
