import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', 
                 learning_rate: float = 2e-5):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            device: 训练设备
            learning_rate: 学习率
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            self.optimizer.zero_grad()
            
            # 将数据移动到指定设备
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(**inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_loader: 评估数据加载器
            
        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc='Evaluating'):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label']
                
                outputs = self.model(**inputs)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              epochs: int) -> List[Dict[str, float]]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            
        Returns:
            每个epoch的训练记录
        """
        history = []
        best_val_f1 = 0
        
        for epoch in range(epochs):
            logging.info(f'Epoch {epoch + 1}/{epochs}')
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调整
            self.scheduler.step()
            
            # 记录训练信息
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **val_metrics
            }
            history.append(metrics)
            
            # 打印训练信息
            logging.info(f'Train Loss: {train_loss:.4f}')
            logging.info(f'Val Metrics: {val_metrics}')
            
            # 保存最佳模型
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'best_model.pt')
            
        return history
