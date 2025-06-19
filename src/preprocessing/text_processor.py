import re
import jieba
from typing import List, Dict, Union

class TextProcessor:
    """文本预处理类"""
    
    def __init__(self, stopwords_path: str = None):
        """
        初始化文本处理器
        
        Args:
            stopwords_path: 停用词文件路径
        """
        self.stopwords = self._load_stopwords(stopwords_path) if stopwords_path else set()
    
    def _load_stopwords(self, path: str) -> set:
        """
        加载停用词表
        
        Args:
            path: 停用词文件路径
            
        Returns:
            停用词集合
        """
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 移除@用户名
        text = re.sub(r'@[\w\-]+', '', text)
        # 移除特殊字符和表情
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def segment(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词列表
        """
        words = jieba.lcut(text)
        # 移除停用词
        if self.stopwords:
            words = [w for w in words if w not in self.stopwords]
        return words
    
    def process(self, text: str) -> List[str]:
        """
        完整的文本处理流程
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的词列表
        """
        # 清理文本
        cleaned_text = self.clean_text(text)
        # 分词
        words = self.segment(cleaned_text)
        return words
