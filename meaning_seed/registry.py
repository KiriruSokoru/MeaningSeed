import json
import os
from typing import Dict, List, Any, Optional

from .i18n import get_t

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
                       scale: float) -> None:
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
