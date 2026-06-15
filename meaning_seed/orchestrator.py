import logging
import os
import json
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_adapter import get_model_adapter
from .i18n import get_t

__all__ = ["Orchestrator"]


class Orchestrator:
    """
    Оркестратор для управления LLM и применения топологических семян.
    
    Класс отвечает за:
    - Загрузку модели и токенизатора
    - Применение масштабирования к мастер-нейронам (хирургия)
    - Сохранение и загрузку семян (JSON-файлов с конфигурацией нейронов)
    - Генерацию текста с помощью модели
    
    Основные методы:
    - apply_scaling(): применение масштабирования к нейронам в указанном слое
    - save_seed(): сохранение конфигурации семени в JSON-файл
    - load_and_apply_seed(): загрузка и применение семени из файла
    - generate(): генерация текста на основе списка сообщений
    """

    def __init__(self, model_name: str, device: str = "auto", lang: str = "ru") -> None:
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.lang = lang
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(get_t("loading_tokenizer_model", lang, model_name=model_name))
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
        self.logger.info(get_t("model_loaded_arch", lang, model_type=self.model.config.model_type))
        self.logger.info(get_t("using_adapter", lang, adapter_name=self.adapter.__class__.__name__))
        self.logger.info(get_t("total_layers", lang, num_layers=self.adapter.get_num_layers(self.model)))

    def get_mlp_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Получить веса MLP слоя по индексу."""
        return self.adapter.get_mlp_weights(self.model, layer_idx)

    def apply_scaling(self, layer_idx: int, master_indices: List[int], scale: float) -> None:
        """Применить масштабирование к мастер-нейронам в указанном слое."""
        self.adapter.scale_master_neurons(self.model, layer_idx, master_indices, scale)
        indices_str = str(master_indices)
        self.logger.info(get_t("scaling_applied", self.lang, scale=scale, layer_idx=layer_idx, indices=indices_str))

    def save_seed(
        self,
        filepath: str,
        layer_idx: int,
        master_indices: List[int],
        scale: float
    ) -> None:
        """
        Сохранить конфигурацию семени в JSON-файл.
        
        Args:
            filepath: Путь к файлу для сохранения
            layer_idx: Индекс слоя, к которому применено масштабирование
            master_indices: Список индексов мастер-нейронов
            scale: Коэффициент масштабирования
        """
        seed_data = {
            "model_name": self.model_name,
            "model_type": self.model.config.model_type,
            "model_hidden_size": getattr(self.model.config, "hidden_size", None),
            "layer_idx": layer_idx,
            "master_indices": master_indices,
            "scale": scale,
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(seed_data, f, indent=2, ensure_ascii=False)
        self.logger.info(get_t("seed_saved", self.lang, filepath=filepath))

    def load_and_apply_seed(self, filepath: str) -> Dict[str, Any]:
        """
        Загрузить семя из файла и применить его к модели.
        
        Args:
            filepath: Путь к JSON-файлу с конфигурацией семени
            
        Returns:
            Словарь с данными загруженного семени
            
        Raises:
            ValueError: Если тип модели в семени не совпадает с текущей моделью
        """
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
        self.logger.info(get_t("seed_applied", self.lang, filepath=filepath))
        return seed_data

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Сгенерировать текст на основе списка сообщений.
        
        Args:
            messages: Список сообщений в формате чата [{"role": "user", "content": "..."}]
            max_new_tokens: Максимальное количество новых токенов для генерации
            temperature: Температура для сэмплирования (выше = более разнообразно)
            top_p: Top-p сэмплирование (nucleus sampling)
            do_sample: Использовать ли сэмплирование (иначе greedy decoding)
            
        Returns:
            Сгенерированный текст (ответ модели)
        """
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
