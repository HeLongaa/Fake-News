import json
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

class DataLoader:
    """数据加载和处理类"""
    
    @staticmethod
    def load_rumors_json(file_path: str) -> List[Dict]:
        """
        加载谣言数据集
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            谣言数据列表
        """
        rumors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                rumors.append(json.loads(line))
        return rumors
    
    @staticmethod
    def load_ced_dataset(base_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        加载CED数据集
        
        Args:
            base_path: CED数据集根目录
            
        Returns:
            原始微博、谣言转发、非谣言转发数据
        """
        # 加载原始微博
        original_path = Path(base_path) / 'original-microblog'
        originals = []
        for file in original_path.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                originals.append(json.load(f))
        
        # 加载谣言转发
        rumor_repost_path = Path(base_path) / 'rumor-repost'
        rumor_reposts = []
        for file in rumor_repost_path.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                rumor_reposts.append(json.load(f))
        
        # 加载非谣言转发
        non_rumor_repost_path = Path(base_path) / 'non-rumor-repost'
        non_rumor_reposts = []
        for file in non_rumor_repost_path.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                non_rumor_reposts.append(json.load(f))
        
        return originals, rumor_reposts, non_rumor_reposts
    
    @staticmethod
    def process_rumors_dataset(rumors: List[Dict]) -> pd.DataFrame:
        """
        处理谣言数据集为DataFrame格式
        
        Args:
            rumors: 谣言数据列表
            
        Returns:
            处理后的DataFrame
        """
        df = pd.DataFrame(rumors)
        # 添加标签列（所有数据都是谣言）
        df['label'] = 1
        return df
    
    @staticmethod
    def process_ced_dataset(originals: List[Dict], 
                          rumor_reposts: List[Dict],
                          non_rumor_reposts: List[Dict]) -> pd.DataFrame:
        """
        处理CED数据集为DataFrame格式
        
        Args:
            originals: 原始微博数据
            rumor_reposts: 谣言转发数据
            non_rumor_reposts: 非谣言转发数据
            
        Returns:
            处理后的DataFrame
        """
        # 处理原始微博
        df_originals = pd.DataFrame(originals)
        
        # 添加标签和转发数据
        df_originals['label'] = df_originals['id'].apply(
            lambda x: 1 if any(r['id'] == x for r in rumor_reposts) else 0
        )
        
        # 添加转发信息
        def get_reposts(post_id):
            for reposts in (rumor_reposts + non_rumor_reposts):
                if reposts['id'] == post_id:
                    return reposts.get('reposts', [])
            return []
            
        df_originals['reposts'] = df_originals['id'].apply(get_reposts)
        
        return df_originals
