import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Callable
from collections import defaultdict
from tqdm import tqdm

from .i18n import get_t

__all__ = ["ActivationExtractor"]


class ActivationExtractor:
    """
    Экстрактор мастер-нейронов через анализ активаций MLP слоёв.
    
    Класс регистрирует хуки на указанные слои модели, собирает активации
    на входных текстах, усредняет их и находит top-k нейронов с максимальной
    активацией (мастер-нейроны).
    
    Основные методы:
    - find_master_neurons(): поиск мастер-нейронов через сбор и анализ активаций
    """

    def __init__(self, model: nn.Module, adapter: Any, lang: str = "ru") -> None:
        """
        Инициализация экстрактора активаций.
        
        Args:
            model: PyTorch модель для анализа
            adapter: Адаптер модели (GPT2Adapter, Qwen2Adapter и т.д.)
            lang: Язык для сообщений ("ru" или "en")
        """
        self.logger = logging.getLogger(__name__)
        self.model: nn.Module = model
        self.adapter: Any = adapter
        self.lang: str = lang
        self._hooks: List[Any] = []
        self._activations: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def _create_hook(self, layer_idx: int) -> Callable:
        """
        Создать hook-функцию для сбора активаций указанного слоя.
        
        Args:
            layer_idx: Индекс слоя для отслеживания активаций
            
        Returns:
            Hook-функция, которая будет вызываться при forward pass
        """
        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            # Отсоединяем от графа вычислений и конвертируем в float32 для стабильности
            act = output.detach().to(torch.float32)
            # Усредняем по batch и sequence dimensions (dim 0 и 1)
            mean_abs_act = act.abs().mean(dim=(0, 1))
            self._activations[layer_idx].append(mean_abs_act)
        
        return hook_fn

    def _remove_all_hooks(self) -> None:
        """Удалить все зарегистрированные хуки из модели."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.logger.debug(f"Removed {len(self._hooks)} hooks from model")

    def find_master_neurons(
        self,
        texts: List[str],
        tokenizer: Any,
        layer_indices: List[int],
        top_k: int = 10,
        batch_size: int = 4
    ) -> Dict[int, List[int]]:
        """
        Найти мастер-нейроны с максимальной активацией в указанных слоях.
        
        Метод регистрирует хуки на указанные слои модели, пропускает через неё
        все тексты батчами, собирает активации, усредняет их и находит top-k
        нейронов с максимальной активацией в каждом слое.
        
        Args:
            texts: Список текстов для анализа активаций
            tokenizer: Токенизатор для преобразования текстов в токены
            layer_indices: Список индексов слоёв для анализа
            top_k: Количество мастер-нейронов для выбора в каждом слое
            batch_size: Размер батча для обработки текстов
            
        Returns:
            Словарь {layer_idx: [neuron_indices]} с индексами мастер-нейронов
            для каждого слоя
            
        Raises:
            ValueError: Если batch_size <= 0
            
        Example:
            >>> extractor = ActivationExtractor(model, adapter)
            >>> master_neurons = extractor.find_master_neurons(
            ...     texts=["Hello world", "Test text"],
            ...     tokenizer=tokenizer,
            ...     layer_indices=[10, 12],
            ...     top_k=10,
            ...     batch_size=2
            ... )
            >>> print(master_neurons)
            {10: [123, 456, 789, ...], 12: [234, 567, 890, ...]}
        """
        # Валидация batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Очистка предыдущих данных
        self._activations.clear()
        self._hooks = []
        
        # Регистрация хуков на все указанные слои
        for layer_idx in layer_indices:
            hook = self.adapter.register_intermediate_hook(
                self.model, 
                layer_idx, 
                self._create_hook(layer_idx)
            )
            self._hooks.append(hook)
        
        self.model.eval()
        
        try:
            # Обработка текстов батчами
            num_batches = (len(texts) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for i in tqdm(
                    range(0, len(texts), batch_size),
                    desc=get_t("collecting_activations", self.lang),
                    unit=get_t("batch", self.lang)
                ):
                    batch_texts = texts[i:i + batch_size]
                    inputs = tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    )
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    _ = self.model(**inputs)
        finally:
            # ГАРАНТИРОВАННОЕ удаление хуков даже при ошибке или прерывании
            self._remove_all_hooks()
        
        # Анализ собранных активаций и поиск мастер-нейронов
        master_neurons: Dict[int, List[int]] = {}
        
        for layer_idx in layer_indices:
            if not self._activations[layer_idx]:
                continue
            
            # Усреднение активаций по всем батчам
            avg_activations = torch.stack(self._activations[layer_idx]).mean(dim=0)
            
            # Поиск top-k нейронов с максимальной активацией
            top_values, top_indices = torch.topk(
                avg_activations, 
                k=min(top_k, len(avg_activations))
            )
            master_neurons[layer_idx] = top_indices.tolist()
            
            self.logger.info(
                get_t("master_neurons_found", self.lang,
                      layer_idx=layer_idx,
                      top_k=top_k,
                      max_act=top_values[0].item())
            )
        
        return master_neurons
