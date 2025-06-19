import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
from typing import List, Dict, Union, Tuple

class RumorDataset(Dataset):
    """谣言数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer: BertTokenizer, max_len: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表 (0: 非谣言, 1: 谣言)
            tokenizer: BERT分词器
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultiModalDataset(Dataset):
    """多模态特征数据集类"""
    
    def __init__(self, texts: List[str], user_features: np.ndarray, 
                 propagation_features: np.ndarray, labels: List[int]):
        """
        初始化数据集
        
        Args:
            texts: 文本特征
            user_features: 用户特征
            propagation_features: 传播特征
            labels: 标签
        """
        self.texts = torch.FloatTensor(texts)
        self.user_features = torch.FloatTensor(user_features)
        self.propagation_features = torch.FloatTensor(propagation_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'text_features': self.texts[idx],
            'user_features': self.user_features[idx],
            'propagation_features': self.propagation_features[idx],
            'label': self.labels[idx]
        }
        
def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        batch_size: 批次大小
        
    Returns:
        训练、验证和测试数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader
