import torch
import torch.nn as nn
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

from .i18n import get_t

class ActivationExtractor:
    def __init__(self, model: nn.Module, adapter, lang: str = "ru"):
        self.model = model
        self.adapter = adapter
        self.lang = lang
        self._hooks = []
        self._activations = defaultdict(list)

    def _create_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            act = output.detach().to(torch.float32)
            mean_abs_act = act.abs().mean(dim=(0, 1)) 
            self._activations[layer_idx].append(mean_abs_act)
        return hook_fn

    def find_master_neurons(self, 
                            texts: List[str], 
                            tokenizer, 
                            layer_indices: List[int], 
                            top_k: int = 10,
                            batch_size: int = 4) -> Dict[int, List[int]]:
        self._activations.clear()
        self._hooks = []

        for layer_idx in layer_indices:
            hook = self.adapter.register_intermediate_hook(self.model, layer_idx, self._create_hook(layer_idx))
            self._hooks.append(hook)

        self.model.eval()
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), 
                         desc=get_t("collecting_activations", self.lang), 
                         unit=get_t("batch", self.lang)):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                _ = self.model(**inputs)

        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        master_neurons = {}
        for layer_idx in layer_indices:
            if not self._activations[layer_idx]:
                continue
            
            avg_activations = torch.stack(self._activations[layer_idx]).mean(dim=0)
            top_values, top_indices = torch.topk(avg_activations, k=min(top_k, len(avg_activations)))
            
            master_neurons[layer_idx] = top_indices.tolist()
            
            print(get_t("master_neurons_found", self.lang, 
                       layer_idx=layer_idx, 
                       top_k=top_k, 
                       max_act=top_values[0].item()))

        return master_neurons
