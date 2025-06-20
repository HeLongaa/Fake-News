import os
import torch
import numpy as np
import time
from datetime import timedelta
from typing import Tuple, List, Union, Optional
import logging

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import cv2
from transformers import BertTokenizer

from Config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
IMAGE_SIZE = 224
TENSOR_DEVICE_TYPE = torch.float32
LABEL_DEVICE_TYPE = torch.long

class MultiModalDataset(Dataset):
    """
    多模态数据集类 - 同时处理文本和图像数据
    
    支持训练模式和推理模式。在训练模式下，返回数据及其标签；
    在推理模式下，仅返回数据。
    """
    
    def __init__(self, path: str, config: Config, is_train: int = 1):
        """
        初始化数据集
        
        Args:
            path: 数据CSV文件路径
            config: 配置对象
            is_train: 是否为训练模式 (1:训练模式, 0:推理模式)
        """
        self.config = config
        self.is_train = is_train
        
        # 加载数据
        df = pd.read_csv(path).sample(frac=self.config.frac)
        self.img_paths = df['path'].to_list()
        self.texts = df['text'].to_list()
        
        # 初始化分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        logger.info(f"已加载BERT分词器: {self.config.bert_name}")
        
        # 训练模式下加载标签
        if self.is_train:
            self.labels = df['label'].to_list()
            logger.info(f"已加载数据集: {len(self.img_paths)}条样本 (训练模式)")
        else:
            logger.info(f"已加载数据集: {len(self.img_paths)}条样本 (推理模式)")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.img_paths)

    def _process_image(self, img_path: str) -> torch.Tensor:
        """
        处理图像数据
        
        Args:
            img_path: 图像文件路径
            
        Returns:
            处理后的图像张量
        """
        try:
            # 读取图像并转换为RGB
            img = Image.open(img_path).convert("RGB")
            
            # 转换为NumPy数组并调整大小
            img = np.array(img)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            
            # 归一化并调整通道顺序
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
            
            # 转换为Tensor
            return torch.tensor(img, dtype=TENSOR_DEVICE_TYPE)
        except Exception as e:
            logger.error(f"处理图像出错 {img_path}: {e}")
            # 返回一个空的图像张量
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=TENSOR_DEVICE_TYPE)

    def _process_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理文本数据
        
        Args:
            text: 输入文本
            
        Returns:
            (input_ids, attention_mask) 的元组
        """
        # 检查文本是否为空或NaN
        if not isinstance(text, str):
            text = ''
        
        # 使用BERT分词器处理文本
        encoded = self.tokenizer(
            text=text, 
            add_special_tokens=True,
            max_length=self.config.pad_size,
            padding='max_length',
            truncation=True
        )
        
        # 转换为Tensor
        input_ids = torch.tensor(encoded['input_ids'], dtype=TENSOR_DEVICE_TYPE)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=TENSOR_DEVICE_TYPE)
        
        return input_ids, attention_mask

    def __getitem__(self, idx: int) -> Union[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], 
                                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            训练模式: ((img, input_ids, attention_mask), label)
            推理模式: (img, input_ids, attention_mask)
        """
        # 处理图像
        img = self._process_image(self.img_paths[idx])
        
        # 处理文本
        input_ids, attention_mask = self._process_text(self.texts[idx])
        
        # 将数据移动到指定设备
        img = img.to(self.config.device)
        input_ids = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        
        # 训练模式下返回数据和标签
        if self.is_train:
            label = torch.tensor(int(self.labels[idx]), dtype=LABEL_DEVICE_TYPE).to(self.config.device)
            return (img, input_ids, attention_mask), label
        
        # 推理模式下仅返回数据
        return (img, input_ids, attention_mask)


def get_time_diff(start_time: float) -> timedelta:
    """
    计算从给定时间点到现在的时间差
    
    Args:
        start_time: 开始时间戳
        
    Returns:
        表示时间差的timedelta对象
    """
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


def create_data_loader(data_path: str, config: Config, batch_size: int = None, 
                      is_train: int = 1, shuffle: bool = True) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据CSV文件路径
        config: 配置对象
        batch_size: 批次大小，默认使用config中的设置
        is_train: 是否为训练模式
        shuffle: 是否打乱数据
        
    Returns:
        数据加载器对象
    """
    if batch_size is None:
        batch_size = config.batch_size
        
    dataset = MultiModalDataset(data_path, config, is_train)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # 可根据系统设置调整
    )


# 测试代码
if __name__ == '__main__':
    config = Config()
    train_data = MultiModalDataset('./data/train.csv', config, 1)
    train_iter = DataLoader(train_data, batch_size=32)
    
    logger.info(f"数据集大小: {len(train_data)}")
    
    # 测试数据加载
    for i, (inputs, label) in enumerate(train_iter):
        if i < 3:  # 只显示前3个批次
            img, input_ids, attention_mask = inputs
            logger.info(f"批次 {i+1}: 图像形状={img.shape}, 文本形状={input_ids.shape}, 标签形状={label.shape}")
        else:
            break