import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from resnet_models import *


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # 构建mask
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        print(logits_mask)
        positives_mask = mask * logits_mask
        print(positives_mask)
        print('*******************')
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

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
class Mynet(nn.Module):
    def __init__(self,config):
        super(Mynet, self).__init__()
        self.config=config
        resnet_name=self.config.resnet_name
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

        self.fc_1 = nn.Linear(self.config.bert_fc+self.config.resnet_fc, self.config.num_classes)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx):
        # BERT
        img,tokens,mask=inx

        img=self.resnet(img)

        outputs = self.bert(tokens,attention_mask=mask)
        pooled_output = outputs[1]
        pooled_output=self.drop(pooled_output)
        fea=torch.cat([img,pooled_output],1)
        logits = self.fc_1(fea)
        logits=self.softmax(logits)

        return img,logits