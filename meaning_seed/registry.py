import json
import os
from typing import Dict, List, Any, Optional, Tuple

from .i18n import get_t


def load_seed(seed_path: str) -> Dict[str, Any]:
    """Загружает и парсит JSON-сид."""
    with open(seed_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_seed_compatibility(seed_data: Dict[str, Any], model_config: Any) -> Tuple[bool, str]:
    """
    Проверяет совместимость сида с конфигурацией модели ДО применения.
    Возвращает кортеж: (is_compatible: bool, error_message: str)
    """
    seed_type = seed_data.get("model_type")
    config_type = getattr(model_config, "model_type", "unknown")
    
    if seed_type and config_type and seed_type != config_type:
        return False, f"Несовпадение типа модели: Сид создан для '{seed_type}', но загружена модель '{config_type}'."
    
    seed_hidden_size = seed_data.get("model_hidden_size")
    config_hidden_size = getattr(model_config, "hidden_size", None)
    
    if seed_hidden_size and config_hidden_size and seed_hidden_size != config_hidden_size:
        return False, (
            f"Критическое несовпадение размерностей: Сид ожидает hidden_size={seed_hidden_size} "
            f"(например, 0.5B), но у загруженной модели hidden_size={config_hidden_size} "
            f"(например, 1.5B). Применение весов невозможно без искажения тензоров."
        )
    
    return True, "Совместимость подтверждена"


def save_seed(seed_data: Dict[str, Any], seed_path: str) -> None:
    """Сохраняет сид с обновленными метаданными."""
    with open(seed_path, 'w', encoding='utf-8') as f:
        json.dump(seed_data, f, indent=2, ensure_ascii=False)


class SeedRegistry:
    def __init__(self, registry_dir: str = "./seeds", lang: str = "ru"):
        self.registry_dir = registry_dir
        self.lang = lang
        os.makedirs(self.registry_dir, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self):
        self._cache = {}
        for filename in os.listdir(self.registry_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.registry_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        proof_name = filename.replace(".json", "")
                        self._cache[proof_name] = data
                    except json.JSONDecodeError:
                        print(get_t("error_reading_file", self.lang, filename=filename))

    def register_proof(self, 
                       proof_name: str, 
                       model_type: str, 
                       layer_idx: int, 
                       master_indices: List[int], 
                       scale: float,
                       model_hidden_size: Optional[int] = None,
                       model_name: Optional[str] = None,
                       scaled_model_path: Optional[str] = None) -> None:
        proof_data = {
            "model_type": model_type,
            "layer_idx": layer_idx,
            "master_indices": master_indices,
            "scale": scale
        }
        if model_hidden_size is not None:
            proof_data["model_hidden_size"] = model_hidden_size
        if model_name is not None:
            proof_data["model_name"] = model_name
        if scaled_model_path is not None:
            proof_data["scaled_model_path"] = scaled_model_path
        self._cache[proof_name] = proof_data
        
        filepath = os.path.join(self.registry_dir, f"{proof_name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(proof_data, f, indent=2)
            
        print(get_t("proof_registered", self.lang, proof_name=proof_name, filepath=filepath))

    def get_proof(self, proof_name: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(proof_name)

    def list_proofs(self) -> List[str]:
        return list(self._cache.keys())

    def is_compatible(self, proof_name: str, current_model_type: str) -> bool:
        proof = self.get_proof(proof_name)
        if not proof:
            return False
        return proof.get("model_type") == current_model_type
