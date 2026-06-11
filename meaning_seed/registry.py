"""
Registry: Сохранение и загрузка топологических семян.
"""

import torch
import os
from typing import Dict, List, Optional
from datetime import datetime


class SeedRegistry:
    """
    Управляет файлами семян.
    Сохраняет семена в формате .pt с метаданными.
    """
    
    @staticmethod
    def save_seed(
        filepath: str,
        model_name: str,
        masters_per_layer: Dict[int, List[int]],
        model: torch.nn.Module,
        task_name: Optional[str] = None
    ) -> Dict:
        """
        Сохраняет семя в файл.
        
        Args:
            filepath: Путь для сохранения (.pt).
            model_name: Имя базовой модели.
            masters_per_layer: Словарь {layer_idx: [master_indices]}.
            model: Модель, из которой извлекаются веса.
            task_name: Имя задачи (для метаданных).
            
        Returns:
            Словарь с метаданными семени.
        """
        seed_data = {
            'model_name': model_name,
            'task_name': task_name or 'unknown',
            'created_at': datetime.now().isoformat(),
            'layers': {}
        }
        
        total_bytes = 0
        for layer_idx, m_list in masters_per_layer.items():
            layer = model.transformer.h[layer_idx].mlp
            
            fc1_w = layer.c_fc.weight.data[:, m_list].clone().cpu()
            fc2_w = layer.c_proj.weight.data[m_list, :].clone().cpu()
            fc1_b = layer.c_fc.bias.data[m_list].clone().cpu()
            
            seed_data['layers'][layer_idx] = {
                'masters': m_list,
                'fc1_w': fc1_w,
                'fc2_w': fc2_w,
                'fc1_b': fc1_b
            }
            
            total_bytes += fc1_w.numel() * 4 + fc2_w.numel() * 4 + fc1_b.numel() * 4
        
        seed_data['seed_size_kb'] = total_bytes / 1024
        seed_data['total_masters'] = sum(len(m) for m in masters_per_layer.values())
        
        torch.save(seed_data, filepath)
        
        print(f"  ✅ Семя сохранено: {filepath}")
        print(f"     Размер: {seed_data['seed_size_kb']:.2f} KB")
        print(f"     Мастеров: {seed_data['total_masters']}")
        
        return seed_data
    
    @staticmethod
    def load_seed(filepath: str) -> Dict:
        """
        Загружает семя из файла.
        
        Returns:
            Словарь с данными семени.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Семя не найдено: {filepath}")
        
        seed_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        print(f"  ✅ Семя загружено: {filepath}")
        print(f"     Задача: {seed_data.get('task_name', 'unknown')}")
        print(f"     Размер: {seed_data.get('seed_size_kb', 0):.2f} KB")
        
        return seed_data
    
    @staticmethod
    def list_seeds(directory: str = "seeds") -> List[str]:
        """Выводит список всех доступных семян в директории."""
        if not os.path.exists(directory):
            return []
        
        return [f for f in os.listdir(directory) if f.endswith('.pt')]
