import torch
import json
import os
from typing import List, Dict, Any, Optional
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

    def save_seed(
        self,
        filepath: str,
        layer_idx: int,
        master_indices: List[int],
        scale: float,
        scaling_factor: Optional[float] = None,
        target_layers: Optional[List[int]] = None,
        neurons: Optional[Dict[int, List[int]]] = None,
        scaled_model_path: Optional[str] = None,
    ) -> None:
        seed_data = {
            "model_name": self.model_name,
            "model_type": self.model.config.model_type,
            "model_hidden_size": getattr(self.model.config, "hidden_size", None),
            "scaling_factor": scaling_factor,
            "target_layers": target_layers,
            "neurons": neurons,
            "scaled_model_path": scaled_model_path,
            "layer_idx": layer_idx,
            "master_indices": master_indices,
            "scale": scale,
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(seed_data, f, indent=2, ensure_ascii=False)
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

    def generate(self, messages: list, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
