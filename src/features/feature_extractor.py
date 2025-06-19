from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """特征提取类"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        
    def extract_text_features(self, texts: List[str]) -> np.ndarray:
        """
        提取文本特征（TF-IDF）
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF特征矩阵
        """
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def extract_user_features(self, user_info: List[Dict[str, Any]]) -> np.ndarray:
        """
        提取用户特征
        
        Args:
            user_info: 用户信息列表
            
        Returns:
            用户特征矩阵
        """
        features = []
        for user in user_info:
            # 提取用户特征，例如：粉丝数、关注数、发文数等
            user_features = [
                user.get('followers_count', 0),
                user.get('friends_count', 0),
                user.get('statuses_count', 0),
                int(user.get('verified', False))
            ]
            features.append(user_features)
        return np.array(features)
    
    def extract_propagation_features(self, 
                                   repost_info: List[List[Dict[str, Any]]]) -> np.ndarray:
        """
        提取传播特征
        
        Args:
            repost_info: 转发信息列表
            
        Returns:
            传播特征矩阵
        """
        features = []
        for reposts in repost_info:
            # 计算传播相关特征
            repost_count = len(reposts)
            time_diffs = []
            depths = []
            
            if repost_count > 0:
                # 计算转发时间差
                times = sorted([pd.to_datetime(r['time']) for r in reposts])
                time_diffs = [(t - times[0]).total_seconds() for t in times]
                
                # 计算转发深度
                for repost in reposts:
                    depth = len(repost.get('repost_path', '').split('/'))
                    depths.append(depth)
            
            # 组合特征
            propagation_features = [
                repost_count,
                np.mean(time_diffs) if time_diffs else 0,
                np.std(time_diffs) if time_diffs else 0,
                max(depths) if depths else 0,
                np.mean(depths) if depths else 0
            ]
            features.append(propagation_features)
            
        return np.array(features)
    
    def combine_features(self, 
                        text_features: np.ndarray, 
                        user_features: np.ndarray, 
                        propagation_features: np.ndarray) -> np.ndarray:
        """
        组合所有特征
        
        Args:
            text_features: 文本特征矩阵
            user_features: 用户特征矩阵
            propagation_features: 传播特征矩阵
            
        Returns:
            组合后的特征矩阵
        """
        return np.hstack([text_features, user_features, propagation_features])
