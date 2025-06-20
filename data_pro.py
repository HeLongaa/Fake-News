import pandas as pd
import os
import csv
import numpy as np
from typing import List, Tuple, Optional
import logging

"""
数据预处理模块

功能:
    1. 读取数据集中的文本和图片路径
    2. 将文本和图片路径写入新的CSV文件中
    3. 划分训练集和验证集
"""

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
DATA_DIR = './data'
IMAGES_DIR = f'{DATA_DIR}/images'
TWEETS_DIR = f'{DATA_DIR}/tweets'
VALIDATION_RATIO = 0.1
CSV_HEADER = ('path', 'text', 'label')


def load_available_images() -> List[str]:
    """加载可用的图片列表"""
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"图片目录 {IMAGES_DIR} 不存在")
        return []
    return os.listdir(IMAGES_DIR)


def process_tweet_data(input_path: str, label: int, output_path: str, available_imgs: List[str]) -> Optional[float]:
    """
    处理推文数据并将结果写入CSV文件
    
    Args:
        input_path: 输入文件路径
        label: 标签 (0=真实, 1=谣言)
        output_path: 输出CSV文件路径
        available_imgs: 可用图片列表
        
    Returns:
        平均句子长度，如果处理失败则返回None
    """
    text_lengths = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        if len(lines) % 3 != 0:
            logger.error(f"文件 {input_path} 的行数不是3的倍数，数据格式可能有误")
            return None
            
        total_entries = len(lines) // 3
        logger.info(f"处理 {input_path} 中的 {total_entries} 条数据")
        
        for i in range(total_entries):
            # 获取元数据、图片URL和文本内容
            _, img_urls, text = lines[i*3], lines[i*3+1].strip(), lines[i*3+2].strip()
            
            # 记录文本长度
            text_lengths.append(len(text))
            
            # 处理图片URL
            img_paths = []
            for img_url in img_urls.split('|'):
                if img_url != 'null':
                    img_name = img_url.split('/')[-1]
                    if img_name in available_imgs:
                        img_paths.append(f'{IMAGES_DIR}/{img_name}')
            
            # 将每个图片与文本和标签组合，写入CSV
            for img_path in img_paths:
                append_to_csv(output_path, [(img_path, text, label)])
                
        avg_length = np.mean(text_lengths) if text_lengths else 0
        logger.info(f"处理完成 {input_path}，平均句子长度: {avg_length:.2f}")
        return avg_length
        
    except Exception as e:
        logger.error(f"处理 {input_path} 时出错: {e}")
        return None


def create_empty_csv(file_path: str) -> None:
    """创建空的CSV文件并写入表头"""
    if os.path.exists(file_path):
        os.remove(file_path)
        
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    
    logger.info(f"创建CSV文件: {file_path}")


def append_to_csv(file_path: str, rows: List[Tuple]) -> None:
    """向CSV文件追加数据行"""
    with open(file_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def split_train_validation(train_path: str, val_ratio: float = VALIDATION_RATIO) -> None:
    """
    将训练集拆分为训练集和验证集
    
    Args:
        train_path: 训练集文件路径
        val_ratio: 验证集比例
    """
    try:
        df = pd.read_csv(train_path, encoding='utf-8')
        val_df = df.sample(frac=val_ratio, random_state=42)  # 固定随机种子以便结果可重现
        train_df = df.drop(index=val_df.index)
        
        logger.info(f"训练集样本数: {len(train_df)}")
        logger.info(f"验证集样本数: {len(val_df)}")
        
        # 保存拆分后的数据集
        val_path = os.path.join(DATA_DIR, 'val.csv')
        val_df.to_csv(val_path, encoding='utf-8', index=None)
        train_df.to_csv(train_path, encoding='utf-8', index=None)
        
        logger.info(f"数据集拆分完成")
    except Exception as e:
        logger.error(f"拆分数据集时出错: {e}")


def main():
    """主函数"""
    # 加载可用图片
    available_imgs = load_available_images()
    logger.info(f"找到 {len(available_imgs)} 张可用图片")
    
    # 处理训练数据
    train_path = os.path.join(DATA_DIR, 'train.csv')
    create_empty_csv(train_path)
    process_tweet_data(os.path.join(TWEETS_DIR, 'train_rumor.txt'), 1, train_path, available_imgs)
    process_tweet_data(os.path.join(TWEETS_DIR, 'train_nonrumor.txt'), 0, train_path, available_imgs)
    
    # 处理测试数据
    test_path = os.path.join(DATA_DIR, 'test.csv')
    create_empty_csv(test_path)
    process_tweet_data(os.path.join(TWEETS_DIR, 'test_rumor.txt'), 1, test_path, available_imgs)
    process_tweet_data(os.path.join(TWEETS_DIR, 'test_nonrumor.txt'), 0, test_path, available_imgs)
    
    # 拆分训练集和验证集
    split_train_validation(train_path)
    
    logger.info("数据预处理完成")


if __name__ == "__main__":
    main()