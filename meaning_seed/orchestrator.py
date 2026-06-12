import torch
import json
import os
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_adapter import get_model_adapter
from .i18n import get_t

class Orchestrator:
    def __init__(self, model_name: str, device: str = "auto", lang: str = "ru"):
        self.model_name = model_name
        self.lang = lang
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(get_t("loading_tokenizer_model", lang, model_name=model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
            
        self.adapter = get_model_adapter(self.model, lang=self.lang)
        
        print(get_t("model_loaded_arch", lang, model_type=self.model.config.model_type))
        print(get_t("using_adapter", lang, adapter_name=self.adapter.__class__.__name__))
        print(get_t("total_layers", lang, num_layers=self.adapter.get_num_layers(self.model)))

    def get_mlp_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        return self.adapter.get_mlp_weights(self.model, layer_idx)

    def apply_scaling(self, layer_idx: int, master_indices: List[int], scale: float) -> None:
        self.adapter.scale_master_neurons(self.model, layer_idx, master_indices, scale)
        indices_str = str(master_indices)
        print(get_t("scaling_applied", self.lang, scale=scale, layer_idx=layer_idx, indices=indices_str))

    def save_seed(self, filepath: str, layer_idx: int, master_indices: List[int], scale: float) -> None:
        seed_data = {
            "model_name": self.model_name,
            "model_type": self.model.config.model_type,
            "layer_idx": layer_idx,
            "master_indices": master_indices,
            "scale": scale
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(seed_data, f, indent=2)
        print(get_t("seed_saved", self.lang, filepath=filepath))

    def load_and_apply_seed(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            seed_data = json.load(f)
        
        if seed_data.get("model_type") != self.model.config.model_type:
            raise ValueError(
                get_t("arch_mismatch", self.lang, 
                      seed_type=seed_data.get("model_type"), 
                      current_type=self.model.config.model_type)
            )
            
        self.apply_scaling(
            seed_data["layer_idx"], 
            seed_data["master_indices"], 
            seed_data["scale"]
        )
        print(get_t("seed_applied", self.lang, filepath=filepath))
        return seed_data
