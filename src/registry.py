import json
import os
from typing import Dict, List, Any, Optional

class SeedRegistry:
    """
    Управляет сохранением и загрузкой "семян" (Real-World Proof).
    Абстрагирован от конкретной модели, полагаясь на метаданные.
    """
    
    def __init__(self, registry_dir: str = "./seeds"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self):
        """Загружает все доступные семена из директории."""
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
                        print(f"⚠️ Ошибка чтения файла: {filename}")

    def register_proof(self, 
                       proof_name: str, 
                       model_type: str, 
                       layer_idx: int, 
                       master_indices: List[int], 
                       scale: float) -> None:
        """Регистрирует новое доказательство в реестре."""
        proof_data = {
            "model_type": model_type,
            "layer_idx": layer_idx,
            "master_indices": master_indices,
            "scale": scale
        }
        self._cache[proof_name] = proof_data
        
        filepath = os.path.join(self.registry_dir, f"{proof_name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(proof_data, f, indent=2)
            
        print(f"✅ Proof '{proof_name}' зарегистрирован и сохранен в {filepath}")

    def get_proof(self, proof_name: str) -> Optional[Dict[str, Any]]:
        """Получает данные доказательства по имени."""
        return self._cache.get(proof_name)

    def list_proofs(self) -> List[str]:
        """Возвращает список имен всех зарегистрированных доказательств."""
        return list(self._cache.keys())

    def is_compatible(self, proof_name: str, current_model_type: str) -> bool:
        """Проверяет совместимость доказательства с текущей архитектурой модели."""
        proof = self.get_proof(proof_name)
        if not proof:
            return False
        return proof.get("model_type") == current_model_type
