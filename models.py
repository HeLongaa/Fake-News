import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from resnet_models import *

class Mynet(nn.Module):
    """
    多模态假新闻检测网络
    
    结合 ResNet 和 BERT 的多模态神经网络，用于处理图像和文本数据。
    
    Args:
        config: 配置对象，包含网络结构的相关参数：
            - resnet_name: ResNet模型类型 ('resnet18', 'resnet34', etc.)
            - resnet_fc: ResNet输出特征维度
            - bert_name: BERT模型名称
            - bert_fc: BERT输出特征维度
            - num_classes: 分类类别数
            - dropout: Dropout比率
    """

    def __init__(self,config):
        super(Mynet, self).__init__()
        self.config=config
        resnet_name=self.config.resnet_name
        #选取resnet种类
        if resnet_name=='resnet18':
            self.resnet=resnet18(self.config.resnet_fc)
        elif resnet_name=='resnet34':
            self.resnet=resnet34(self.config.resnet_fc)
        elif resnet_name=='resnet50':
            self.resnet=resnet50(self.config.resnet_fc)
        elif resnet_name=='resnet101':
            self.resnet=resnet101(self.config.resnet_fc)
        elif resnet_name=='resnet152':
            self.resnet=resnet152(self.config.resnet_fc)

        self.bert= BertModel.from_pretrained(self.config.bert_name)
        #bert的种类

        self.fc_1 = nn.Linear(self.config.bert_fc+self.config.resnet_fc, self.config.num_classes)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx):
        """
        前向传播函数
        
        Args:
            inx: 包含三个元素的元组 (img, tokens, mask)：
                - img: 图像输入张量，形状为 (batch_size, 3, H, W)
                - tokens: BERT输入token序列，形状为 (batch_size, seq_len)
                - mask: BERT注意力掩码，形状为 (batch_size, seq_len)
                
        Returns:
            tuple：包含两个元素：
                - img: ResNet提取的图像特征
                - logits: 分类输出概率，经过softmax处理
        """
        # BERT
        img,tokens,mask=inx

        # attention_mask=mask
        img=self.resnet(img)

        outputs = self.bert(tokens,attention_mask=mask)
        # emb (32,128)-(32,768)
        pooled_output = outputs[1]
        pooled_output=self.drop(pooled_output)
        fea=torch.cat([img,pooled_output],1)
        # fea=self.drop(fea)
        logits = self.fc_1(fea)
        logits=self.softmax(logits)
        # 返回的第一个是需要对比的特征，img就为图像特征，fea就为全特征
        return img,logits
   
