import torch
import json
import os
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_adapter import get_model_adapter

class Orchestrator:
    """
    Главный координатор для работы с моделью: загрузка, извлечение весов, 
    применение масштабирования и управление семенами (seeds).
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Загрузка токенизатора и модели: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Используем AutoModel для поддержки любой архитектуры
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda": # Если device_map="auto" не использовался, переносим вручную
            self.model = self.model.to(self.device)
            
        # Инициализируем правильный адаптер на основе конфигурации загруженной модели
        self.adapter = get_model_adapter(self.model)
        
        print(f"✅ Модель загружена. Архитектура: {self.model.config.model_type}")
        print(f"🔧 Используется адаптер: {self.adapter.__class__.__name__}")
        print(f"📊 Всего слоёв: {self.adapter.get_num_layers(self.model)}")

    def get_mlp_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Получает веса MLP указанного слоя через адаптер."""
        return self.adapter.get_mlp_weights(self.model, layer_idx)

    def apply_scaling(self, layer_idx: int, master_indices: List[int], scale: float) -> None:
        """Применяет масштабирование к мастер-нейронам через адаптер."""
        self.adapter.scale_master_neurons(self.model, layer_idx, master_indices, scale)
        print(f"⚡ Масштабирование x{scale} применено к слою {layer_idx}, нейроны: {master_indices}")

    def save_seed(self, filepath: str, layer_idx: int, master_indices: List[int], scale: float) -> None:
        """Сохраняет семя (Real-World Proof) в JSON файл."""
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
        print(f"💾 Семя сохранено в {filepath}")

    def load_and_apply_seed(self, filepath: str) -> Dict[str, Any]:
        """Загружает семя из файла и применяет его к текущей модели."""
        with open(filepath, 'r', encoding='utf-8') as f:
            seed_data = json.load(f)
        
        # Проверка совместимости архитектур
        if seed_data.get("model_type") != self.model.config.model_type:
            raise ValueError(
                f"❌ Несоответствие архитектур! Семя создано для: {seed_data.get('model_type')}, "
                f"текущая модель: {self.model.config.model_type}"
            )
            
        self.apply_scaling(
            seed_data["layer_idx"], 
            seed_data["master_indices"], 
            seed_data["scale"]
        )
        print(f"✅ Семя из {filepath} успешно применено.")
        return seed_data
