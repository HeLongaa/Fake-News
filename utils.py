import os
import torch
import numpy as np
import time
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import cv2
from transformers import BertTokenizer
from Config import Config
from typing import Tuple, List, Union, Optional

"""
数据加载和预处理模块

功能:
    1. 多模态数据集加载类(My_Dataset)：处理文本和图像数据
    2. 时间差计算函数(get_time_dif)：计算运行时间
"""

class My_Dataset(Dataset):
    """
    多模态数据加载类，同时处理文本和图像数据
    
    属性:
        config: 配置对象，包含模型参数
        iftrain: 标识是否为训练模式，1表示训练模式，0表示测试模式
        img_path: 图像文件路径列表
        text: 文本内容列表
        labels: 标签列表（仅在训练模式下有效）
        tokenizer: BERT分词器
    """
    
    def __init__(self, path: str, config: Config, iftrain: int):
        """
        初始化数据集
        
        参数:
            path: 数据集CSV文件路径
            config: 配置对象
            iftrain: 是否为训练模式（1=训练，0=测试）
        """
        self.config = config
        self.iftrain = iftrain
        
        # 加载数据集并按配置的比例采样
        df = pd.read_csv(path).sample(frac=self.config.frac)
        self.img_path = df['path'].to_list()
        self.text = df['text'].to_list()
        
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        
        # 训练模式下加载标签
        if self.iftrain == 1:
            self.labels = df['label'].to_list()
    def __getitem__(self, idx: int) -> Union[Tuple, Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            训练模式: ((图像张量, 文本输入ID张量, 注意力掩码张量), 标签张量)
            测试模式: (图像张量, 文本输入ID张量, 注意力掩码张量)
        """
        # 处理图像数据
        img = self._process_image(self.img_path[idx])
        
        # 处理文本数据
        text = self.text[idx]
        input_id, attention_mask = self._process_text(text)
        
        # 根据模式返回不同格式的数据
        if self.iftrain == 1:
            label = torch.tensor(int(self.labels[idx]), dtype=torch.long).to(self.config.device)
            return (img, input_id, attention_mask), label
        else:
            return (img, input_id, attention_mask)
    
    def _process_image(self, img_path: str) -> torch.Tensor:
        """
        处理图像数据
        
        参数:
            img_path: 图像文件路径
            
        返回:
            处理后的图像张量
        """
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (224, 224))  # 调整大小