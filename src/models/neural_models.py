import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTClassifier(nn.Module):
    """基于BERT的虚假新闻分类器"""
    
    def __init__(self, pretrained_model: str = 'bert-base-chinese'):
        """
        初始化BERT分类器
        
        Args:
            pretrained_model: 预训练模型名称
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # 二分类
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入序列的标记ID
            attention_mask: 注意力掩码
            
        Returns:
            分类logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

class BiLSTMAttention(nn.Module):
    """带注意力机制的BiLSTM模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int):
        """
        初始化BiLSTM+Attention模型
        
        Args:
            vocab_size: 词表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
        """
        super(BiLSTMAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 二分类
        
    def attention_net(self, lstm_output: torch.Tensor, 
                     final_state: torch.Tensor) -> torch.Tensor:
        """
        注意力层
        
        Args:
            lstm_output: LSTM所有时间步的输出
            final_state: LSTM最终状态
            
        Returns:
            加权后的特征向量
        """
        hidden = final_state.view(-1, 1, 1)
        attention_weights = torch.tanh(self.attention(lstm_output))
        soft_attention_weights = torch.softmax(attention_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attention_weights).squeeze(2)
        return context
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            text: 输入文本序列
            
        Returns:
            分类logits
        """
        embedded = self.embedding(text)
        lstm_output, (final_hidden_state, _) = self.lstm(embedded)
        
        attention_output = self.attention_net(lstm_output, final_hidden_state)
        logits = self.fc(attention_output)
        
        return logits
