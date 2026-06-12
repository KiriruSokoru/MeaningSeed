import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable

class BaseModelAdapter(ABC):
    """Базовый абстрактный класс адаптера модели."""
    
    @abstractmethod
    def get_num_layers(self, model: nn.Module) -> int:
        pass

    @abstractmethod
    def get_mlp_input_dim(self, model: nn.Module, layer_idx: int) -> int:
        pass

    @abstractmethod
    def get_mlp_hidden_dim(self, model: nn.Module, layer_idx: int) -> int:
        pass

    @abstractmethod
    def get_mlp_weights(self, model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def set_mlp_weights(self, model: nn.Module, layer_idx: int, weights: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def scale_master_neurons(self, model: nn.Module, layer_idx: int, master_indices: List[int], scale: float) -> None:
        """Применяет масштабирование к весам мастер-нейронов."""
        pass

    @abstractmethod
    def register_intermediate_hook(self, model: nn.Module, layer_idx: int, hook_fn: Callable) -> Any:
        """
        Регистрирует хук на промежуточном слое MLP (до проекции down), 
        чтобы собирать активации в расширенном пространстве (hidden_dim).
        Возвращает handle для удаления хука.
        """
        pass


class GPT2Adapter(BaseModelAdapter):
    """Адаптер для архитектур семейства GPT-2 (включая distilgpt2)."""
    
    def get_num_layers(self, model: nn.Module) -> int:
        return len(model.transformer.h)

    def get_mlp_input_dim(self, model: nn.Module, layer_idx: int) -> int:
        return model.config.n_embd

    def get_mlp_hidden_dim(self, model: nn.Module, layer_idx: int) -> int:
        return model.transformer.h[layer_idx].mlp.c_fc.out_features

    def get_mlp_weights(self, model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
        mlp = model.transformer.h[layer_idx].mlp
        return {
            "up": mlp.c_fc.weight.data,
            "down": mlp.c_proj.weight.data
        }

    def set_mlp_weights(self, model: nn.Module, layer_idx: int, weights: Dict[str, torch.Tensor]) -> None:
        mlp = model.transformer.h[layer_idx].mlp
        if "up" in weights:
            mlp.c_fc.weight.data = weights["up"].to(mlp.c_fc.weight.device)
        if "down" in weights:
            mlp.c_proj.weight.data = weights["down"].to(mlp.c_proj.weight.device)

    def scale_master_neurons(self, model: nn.Module, layer_idx: int, master_indices: List[int], scale: float) -> None:
        mlp = model.transformer.h[layer_idx].mlp
        # c_fc (up): форма (hidden_dim, input_dim). Масштабируем столбцы (dim=1)
        mlp.c_fc.weight.data[:, master_indices] *= scale
        # c_proj (down): форма (input_dim, hidden_dim). Масштабируем строки (dim=0)
        mlp.c_proj.weight.data[master_indices, :] *= scale

    def register_intermediate_hook(self, model: nn.Module, layer_idx: int, hook_fn: Callable) -> Any:
        # Хук на c_fc дает нам активации размерности (batch, seq, hidden_dim)
        return model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(hook_fn)


class Qwen2Adapter(BaseModelAdapter):
    """Адаптер для архитектур семейства Qwen2 / Qwen2.5."""
    
    def get_num_layers(self, model: nn.Module) -> int:
        return model.config.num_hidden_layers

    def get_mlp_input_dim(self, model: nn.Module, layer_idx: int) -> int:
        return model.config.hidden_size

    def get_mlp_hidden_dim(self, model: nn.Module, layer_idx: int) -> int:
        return model.config.intermediate_size

    def get_mlp_weights(self, model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
        mlp = model.model.layers[layer_idx].mlp
        return {
            "gate": mlp.gate_proj.weight.data,
            "up": mlp.up_proj.weight.data,
            "down": mlp.down_proj.weight.data
        }

    def set_mlp_weights(self, model: nn.Module, layer_idx: int, weights: Dict[str, torch.Tensor]) -> None:
        mlp = model.model.layers[layer_idx].mlp
        if "gate" in weights:
            mlp.gate_proj.weight.data = weights["gate"].to(mlp.gate_proj.weight.device)
        if "up" in weights:
            mlp.up_proj.weight.data = weights["up"].to(mlp.up_proj.weight.device)
        if "down" in weights:
            mlp.down_proj.weight.data = weights["down"].to(mlp.down_proj.weight.device)

    def scale_master_neurons(self, model: nn.Module, layer_idx: int, master_indices: List[int], scale: float) -> None:
        mlp = model.model.layers[layer_idx].mlp
        # gate_proj и up_proj: форма (hidden_dim, input_dim). Масштабируем столбцы (dim=1)
        mlp.gate_proj.weight.data[:, master_indices] *= scale
        mlp.up_proj.weight.data[:, master_indices] *= scale
        # down_proj: форма (input_dim, hidden_dim). Масштабируем строки (dim=0)
        mlp.down_proj.weight.data[master_indices, :] *= scale

    def register_intermediate_hook(self, model: nn.Module, layer_idx: int, hook_fn: Callable) -> Any:
        # В Qwen2 (SwiGLU) промежуточная активация формируется как silu(gate) * up.
        # Выход up_proj имеет нужную нам размерность (intermediate_size) и отлично подходит 
        # как прокси для оценки величины активации нейрона в этом слое.
        return model.model.layers[layer_idx].mlp.up_proj.register_forward_hook(hook_fn)


def get_model_adapter(model: nn.Module) -> BaseModelAdapter:
    """Фабричная функция для получения правильного адаптера."""
    model_type = getattr(model.config, "model_type", "").lower()
    
    if "gpt2" in model_type:
        return GPT2Adapter()
    elif "qwen2" in model_type:
        return Qwen2Adapter()
    else:
        raise ValueError(f"Неподдерживаемая архитектура модели: {model_type}. "
                         f"Поддерживаются: gpt2, qwen2")
