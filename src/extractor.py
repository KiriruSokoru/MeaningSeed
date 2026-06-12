import torch
import torch.nn as nn
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

class ActivationExtractor:
    """
    Извлекает активации MLP для определения "мастер-нейронов" 
    (нейронов с наибольшей средней абсолютной активацией).
    """
    
    def __init__(self, model: nn.Module, adapter):
        self.model = model
        self.adapter = adapter
        self._hooks = []
        self._activations = defaultdict(list)

    def _create_hook(self, layer_idx: int):
        """Создает функцию хука для конкретного слоя."""
        def hook_fn(module, input, output):
            # output имеет форму (batch, seq_len, intermediate_hidden_dim)
            # Нам нужно усреднить по batch и seq_len, чтобы получить важность каждого нейрона
            # Приводим к float32 для точности накопления
            act = output.detach().to(torch.float32)
            # Усредняем абсолютные значения по батчу и длине последовательности
            mean_abs_act = act.abs().mean(dim=(0, 1)) 
            self._activations[layer_idx].append(mean_abs_act)
        return hook_fn

    def find_master_neurons(self, 
                            texts: List[str], 
                            tokenizer, 
                            layer_indices: List[int], 
                            top_k: int = 10,
                            batch_size: int = 4) -> Dict[int, List[int]]:
        """
        Прогоняет тексты через модель и находит top_k наиболее активных нейронов для указанных слоев.
        """
        self._activations.clear()
        self._hooks = []

        # 1. Регистрируем хуки через адаптер
        for layer_idx in layer_indices:
            hook = self.adapter.register_intermediate_hook(self.model, layer_idx, self._create_hook(layer_idx))
            self._hooks.append(hook)

        # 2. Собираем активации
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="🧠 Сбор активаций MLP", unit="batch"):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                
                # Переносим входы на то же устройство, где находится модель
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                _ = self.model(**inputs)

        # 3. Удаляем хуки, чтобы не засорять память
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        # 4. Агрегируем результаты и находим топ-K
        master_neurons = {}
        for layer_idx in layer_indices:
            if not self._activations[layer_idx]:
                continue
            
            # Усредняем активации по всем батчам
            avg_activations = torch.stack(self._activations[layer_idx]).mean(dim=0)
            
            # Находим индексы топ-K максимальных значений
            top_values, top_indices = torch.topk(avg_activations, k=min(top_k, len(avg_activations)))
            
            master_neurons[layer_idx] = top_indices.tolist()
            
            print(f"🔍 Слой {layer_idx}: Топ-{top_k} мастер-нейронов найдены. "
                  f"Макс. активация: {top_values[0].item():.4f}")

        return master_neurons
