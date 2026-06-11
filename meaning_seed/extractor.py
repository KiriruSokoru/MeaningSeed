"""
Extractor: Поиск топологических мастеров через кривизну Оливье-Риччи.
"""

import warnings
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from typing import Dict, List, Optional


class MeaningExtractor:
    """
    Извлекает топологических мастеров из модели.
    Использует Degree Capping для оптимизации расчета Риччи на CPU.
    """
    
    def __init__(
        self,
        max_degree: int = 15,
        corr_threshold: float = 0.4,
        activation_samples: int = 10,
        device: Optional[torch.device] = None
    ):
        self.max_degree = max_degree
        self.corr_threshold = corr_threshold
        self.activation_samples = activation_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_distributed_masters(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        masters_per_layer: int = 10,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, List[int]]:
        """
        Извлекает мастеров из каждого слоя модели (распределенное семя).
        
        Args:
            model: Обученная модель (GPT-2 или совместимая).
            dataloader: Даталоадер с данными задачи.
            masters_per_layer: Сколько мастеров извлекать с каждого слоя.
            layer_indices: Какие слои анализировать. По умолчанию - все.
            
        Returns:
            Словарь {layer_idx: [master_indices]}.
        """
        model.eval()
        num_layers = len(model.transformer.h)
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        distributed_masters = {}
        
        pbar_layers = tqdm(layer_indices, desc="  Извлечение мастеров", leave=True)
        for layer_idx in pbar_layers:
            pbar_layers.set_description(f"  Анализ слоя {layer_idx}/{num_layers-1}")
            
            # 1. Сбор активаций
            acts = self._collect_activations(model, dataloader, layer_idx)
            
            # 2. Построение графа корреляций с Degree Capping
            G = self._build_capped_graph(acts)
            
            # 3. Расчет кривизны Риччи
            curv = self._compute_ricci_curvature(G)
            
            # 4. Выбор топ-K мастеров
            sorted_m = sorted(curv.items(), key=lambda x: x[1], reverse=True)
            distributed_masters[layer_idx] = [n for n, c in sorted_m[:masters_per_layer]]
        
        return distributed_masters
    
    def _collect_activations(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer_idx: int
    ) -> np.ndarray:
        """Собирает активации MLP-слоя (c_fc) для анализа."""
        acts = []
        
        def hook_fn(module, input, output):
            # Усредняем по seq_len для получения стабильного признака нейрона
            acts.append(output.mean(dim=1).detach().cpu().numpy())
        
        target_layer = model.transformer.h[layer_idx].mlp.c_fc
        handle = target_layer.register_forward_hook(hook_fn)
        
        pbar_act = tqdm(dataloader, desc="    Сбор активаций", leave=False)
        with torch.no_grad():
            for batch in pbar_act:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                model(input_ids=input_ids, attention_mask=attention_mask)
                if len(acts) >= self.activation_samples:
                    break
        handle.remove()
        
        return np.concatenate(acts, axis=0)
    
    def _build_capped_graph(self, acts: np.ndarray) -> nx.Graph:
        """
        Строит разреженный граф корреляций с ограничением степени узла.
        Это ускоряет расчет Риччи на порядки.
        """
        num_nodes = acts.shape[1]
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.nan_to_num(np.abs(np.corrcoef(acts.T)), nan=0.0)
        
        capped_edges = []
        pbar_graph = tqdm(range(num_nodes), desc="    Построение графа", leave=False)
        
        for i in pbar_graph:
            row_corr = corr[i, i+1:]
            if len(row_corr) == 0:
                continue
            top_idx = np.argsort(row_corr)[-self.max_degree:]
            for idx in top_idx:
                j = i + 1 + idx
                weight = row_corr[idx]
                if weight > self.corr_threshold:
                    capped_edges.append((i, j, weight))
        
        G.add_weighted_edges_from(capped_edges)
        
        if len(capped_edges) == 0:
            # Fallback: если граф пуст, берем случайные узлы
            return G
        
        print(f"    Граф: {num_nodes} узлов, {len(capped_edges)} ребер")
        return G
    
    def _compute_ricci_curvature(self, G: nx.Graph) -> Dict[int, float]:
        """Вычисляет кривизну Оливье-Риччи для всех узлов графа."""
        if len(G.edges) == 0:
            return {n: 0.0 for n in G.nodes}
        
        orc = OllivierRicci(G, alpha=0.5)
        Gr = orc.compute_ricci_curvature()
        return {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
