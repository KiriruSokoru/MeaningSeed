"""
Orchestrator: Ядро оркестрации задач - inject, eject, targeted_warmup.
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Orchestrator:
    """
    Управляет базовой моделью-нодой.
    Позволяет мгновенно внедрять и извлекать топологические семена.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Загрузка базовой модели {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Сохраняем "чистые" веса для восстановления
        self._base_weights_backup = {}
        self._active_seed = None
    
    def inject_seed(
        self,
        seed_data: Dict,
        warmup_epochs: int = 1,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        lr: float = 1e-3
    ) -> None:
        """
        Внедряет семя в модель и выполняет целевой прогрев.
        
        Args:
            seed_data: Словарь с весами мастеров (из registry.load_seed).
            warmup_epochs: Сколько эпох прогревать (обновляя только мастеров).
            dataloader: Нужен для warmup. Если None - пропускает прогрев.
            lr: Learning rate для warmup.
        """
        # Сохраняем базовые веса, если еще не сохранены
        if not self._base_weights_backup:
            self._backup_base_weights(seed_data)
        
        # Если есть активное семя - сначала eject
        if self._active_seed is not None:
            self.eject_seed()
        
        # Внедряем веса мастеров
        print("  Inject: внедрение топологических якорей...")
        for layer_idx, layer_data in seed_data['layers'].items():
            layer_idx = int(layer_idx)
            m_list = layer_data['masters']
            layer = self.model.transformer.h[layer_idx].mlp
            
            for local_idx, m_idx in enumerate(m_list):
                layer.c_fc.weight.data[:, m_idx] = layer_data['fc1_w'][:, local_idx].to(self.device)
                layer.c_proj.weight.data[m_idx, :] = layer_data['fc2_w'][local_idx, :].to(self.device)
                layer.c_fc.bias.data[m_idx] = layer_data['fc1_b'][local_idx].to(self.device)
        
        self._active_seed = seed_data
        
        # Targeted Warmup (только мастера)
        if warmup_epochs > 0 and dataloader is not None:
            print(f"  Warmup: {warmup_epochs} эпоха(и) целевого прогрева...")
            self._targeted_warmup(dataloader, seed_data, warmup_epochs, lr)
    
    def eject_seed(self) -> None:
        """
        Деструктивное извлечение: заменяет веса мастеров на шум.
        Возвращает модель в "чистое" состояние.
        """
        if self._active_seed is None:
            print("  Нода уже чиста, нечего извлекать.")
            return
        
        print("  Eject: замена якорей на шум...")
        for layer_idx, layer_data in self._active_seed['layers'].items():
            layer_idx = int(layer_idx)
            m_list = layer_data['masters']
            layer = self.model.transformer.h[layer_idx].mlp
            
            for m_idx in m_list:
                # Возвращаем оригинальные базовые веса
                layer.c_fc.weight.data[:, m_idx] = self._base_weights_backup[layer_idx][m_idx]['fc1_w'].to(self.device)
                layer.c_proj.weight.data[m_idx, :] = self._base_weights_backup[layer_idx][m_idx]['fc2_w'].to(self.device)
                layer.c_fc.bias.data[m_idx] = self._base_weights_backup[layer_idx][m_idx]['fc1_b'].to(self.device)
        
        self._active_seed = None
    
    def _backup_base_weights(self, seed_data: Dict) -> None:
        """Сохраняет базовые веса для всех мастеров, которые будут задействованы."""
        print("  Backup: сохранение базовых весов ноды...")
        for layer_idx, layer_data in seed_data['layers'].items():
            layer_idx = int(layer_idx)
            m_list = layer_data['masters']
            layer = self.model.transformer.h[layer_idx].mlp
            
            self._base_weights_backup[layer_idx] = {}
            for m_idx in m_list:
                self._base_weights_backup[layer_idx][m_idx] = {
                    'fc1_w': layer.c_fc.weight.data[:, m_idx].clone().cpu(),
                    'fc2_w': layer.c_proj.weight.data[m_idx, :].clone().cpu(),
                    'fc1_b': layer.c_fc.bias.data[m_idx].clone().cpu()
                }
    
    def _targeted_warmup(
        self,
        dataloader: torch.utils.data.DataLoader,
        seed_data: Dict,
        epochs: int,
        lr: float
    ) -> None:
        """
        Целевой прогрев: обновляет ТОЛЬКО веса мастеров.
        Остальные нейроны остаются нетронутыми (градиенты зануляются маской).
        """
        # Замораживаем всё
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Размораживаем все MLP-слои (но градиенты будем маскировать)
        for i in range(len(self.model.transformer.h)):
            self.model.transformer.h[i].mlp.c_fc.weight.requires_grad = True
            self.model.transformer.h[i].mlp.c_proj.weight.requires_grad = True
            self.model.transformer.h[i].mlp.c_fc.bias.requires_grad = True
        
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        
        pbar_epochs = tqdm(range(epochs), desc="    Targeted Warmup", leave=True)
        for epoch in pbar_epochs:
            for batch in dataloader:
                opt.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids.clone()).to(self.device)
                
                out = self.model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                
                # Маскируем градиенты: оставляем только для мастеров
                for layer_idx, layer_data in seed_data['layers'].items():
                    layer_idx = int(layer_idx)
                    m_list = layer_data['masters']
                    layer = self.model.transformer.h[layer_idx].mlp
                    
                    mask_fc1 = torch.zeros_like(layer.c_fc.weight.grad)
                    mask_fc2 = torch.zeros_like(layer.c_proj.weight.grad)
                    mask_bias = torch.zeros_like(layer.c_fc.bias.grad)
                    
                    for m_idx in m_list:
                        mask_fc1[:, m_idx] = 1
                        mask_fc2[m_idx, :] = 1
                        mask_bias[m_idx] = 1
                    
                    layer.c_fc.weight.grad.mul_(mask_fc1)
                    layer.c_proj.weight.grad.mul_(mask_fc2)
                    layer.c_fc.bias.grad.mul_(mask_bias)
                
                opt.step()
            
            pbar_epochs.set_postfix({'loss': f"{out.loss.item():.3f}"})
    
    def evaluate_perplexity(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Оценивает перплексию модели на даталоадере."""
        self.model.eval()
        total_loss, total_batches = 0, 0
        
        pbar = tqdm(dataloader, desc="  Оценка", leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids.clone()).to(self.device)
                out = self.model(input_ids=input_ids, labels=labels)
                total_loss += out.loss.item()
                total_batches += 1
        
        self.model.train()
        return float(np.exp(total_loss / total_batches))
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Генерирует текст (для быстрой проверки навыка)."""
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
