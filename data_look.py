import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from typing import Dict, List
import logging
import matplotlib as mpl


import os
import logging
from typing import Dict, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_samples_in_file(file_path: str) -> int:
    """
    统计文本文件中的样本数量
    每三行为一个样本
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return len(lines) // 3
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return 0

def analyze_dataset_statistics() -> Dict[str, Tuple[int, float, float]]:
    """
    分析数据集统计信息
    返回：数据集名称 -> (总样本数, 真实新闻比例, 虚假新闻比例)
    """
    # 定义文件路径
    data_dir = './data'
    tweets_dir = os.path.join(data_dir, 'tweets')
    
    # 统计各个文件的样本数
    train_rumor = count_samples_in_file(os.path.join(tweets_dir, 'train_rumor.txt'))
    train_nonrumor = count_samples_in_file(os.path.join(tweets_dir, 'train_nonrumor.txt'))
    test_rumor = count_samples_in_file(os.path.join(tweets_dir, 'test_rumor.txt'))
    test_nonrumor = count_samples_in_file(os.path.join(tweets_dir, 'test_nonrumor.txt'))
    
    # 计算训练集和测试集的统计信息
    train_total = train_rumor + train_nonrumor
    test_total = test_rumor + test_nonrumor
    
    # 验证集大小（根据readme，原始数据没有验证集，将从训练集中划分10%）
    val_total = int(train_total * 0.1)
    train_total = train_total - val_total
    
    # 计算各个数据集的比例
    train_real_ratio = (train_nonrumor * 0.9 / train_total) * 100
    train_fake_ratio = (train_rumor * 0.9 / train_total) * 100
    
    val_real_ratio = (train_nonrumor * 0.1 / val_total) * 100
    val_fake_ratio = (train_rumor * 0.1 / val_total) * 100
    
    test_real_ratio = (test_nonrumor / test_total) * 100
    test_fake_ratio = (test_rumor / test_total) * 100
    
    # 生成LaTeX代码
    latex_code = f"""
\\begin{{table}}[htbp]
    \\centering
    \\caption{{数据集统计信息}}
    \\label{{tab:dataset_stats}}
    \\begin{{tabular}}{{lccc}}
        \\toprule
        \\textbf{{数据集}} & \\textbf{{样本数}} & \\textbf{{真实新闻比例}} & \\textbf{{虚假新闻比例}} \\\\
        \\midrule
        训练集 & {train_total} & {train_real_ratio:.2f}\\% & {train_fake_ratio:.2f}\\% \\\\
        验证集 & {val_total} & {val_real_ratio:.2f}\\% & {val_fake_ratio:.2f}\\% \\\\
        测试集 & {test_total} & {test_real_ratio:.2f}\\% & {test_fake_ratio:.2f}\\% \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
    
    # 打印统计信息
    print("\n数据集统计信息：")
    print(f"训练集：{train_total} 样本")
    print(f"  - 真实新闻：{train_real_ratio:.2f}%")
    print(f"  - 虚假新闻：{train_fake_ratio:.2f}%")
    print(f"\n验证集：{val_total} 样本")
    print(f"  - 真实新闻：{val_real_ratio:.2f}%")
    print(f"  - 虚假新闻：{val_fake_ratio:.2f}%")
    print(f"\n测试集：{test_total} 样本")
    print(f"  - 真实新闻：{test_real_ratio:.2f}%")
    print(f"  - 虚假新闻：{test_fake_ratio:.2f}%")
    
    # 输出LaTeX代码
    print("\nLaTeX代码：")
    print(latex_code)
    
    return {
        '训练集': (train_total, train_real_ratio, train_fake_ratio),
        '验证集': (val_total, val_real_ratio, val_fake_ratio),
        '测试集': (test_total, test_real_ratio, test_fake_ratio)
    }


'''
处理后的数据统计
'''
# 配置中文字体，使用更通用的字体设置
def setup_chinese_font():
    """设置中文字体，按优先级尝试不同的字体"""
    fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    for font in fonts:
        try:
            mpl.rc('font', family=font)
            mpl.rcParams['axes.unicode_minus'] = False
            # 测试字体是否可用
            plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试', fontsize=12)
            plt.close()
            print(f"成功设置字体: {font}")
            return True
        except:
            continue
    
    # 如果没有合适的中文字体，使用默认字体并打印警告
    print("警告：未找到合适的中文字体，图表中的中文可能无法正常显示")
    return False

class DatasetAnalyzer:
    def __init__(self, data_dir: str):
        """初始化数据集分析器"""
        self.data_dir = data_dir
        self.datasets = {
            'Training': 'train.csv',
            'Validation': 'val.csv',
            'Testing': 'test.csv'
        }
        self.label_names = {0: 'Real', 1: 'Fake'}
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据集"""
        return {
            name: pd.read_csv(os.path.join(self.data_dir, path))
            for name, path in self.datasets.items()
        }
        
    def analyze_distribution(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """分析数据集的类别分布"""
        plt.figure(figsize=(10, 6))
        
        data = []
        labels = []
        categories = []
        
        for name, df in dfs.items():
            counts = df['label'].value_counts()
            for label, count in counts.items():
                data.append(count)
                labels.append(self.label_names[label])
                categories.append(name)
        
        df_plot = pd.DataFrame({
            'Dataset': categories,
            'Category': labels,
            'Count': data
        })
        
        sns.barplot(x='Dataset', y='Count', hue='Category', data=df_plot)
        plt.title('Class Distribution in Datasets')
        plt.savefig('assets/dataset_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def analyze_text_length(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """分析文本长度分布"""
        plt.figure(figsize=(12, 6))
        
        for name, df in dfs.items():
            text_lengths = df['text'].str.len()
            sns.kdeplot(data=text_lengths, label=name)
            
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('assets/text_length_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def print_statistics(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """打印数据集统计信息"""
        print("\nDataset Statistics:")
        print("=" * 50)
        
        for name, df in dfs.items():
            print(f"\n{name} Set Statistics:")
            print(f"Total samples: {len(df)}")
            print("\nClass distribution:")
            for label, count in df['label'].value_counts().items():
                print(f"{self.label_names[label]}: {count} ({count/len(df)*100:.2f}%)")
            
            text_lengths = df['text'].str.len()
            print(f"\nText length statistics:")
            print(f"Mean length: {text_lengths.mean():.2f}")
            print(f"Max length: {text_lengths.max()}")
            print(f"Min length: {text_lengths.min()}")
            print(f"Median length: {text_lengths.median()}")
            print("=" * 50)
            
    def analyze_image_distribution(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """分析每条新闻平均包含的图片数"""
        plt.figure(figsize=(10, 6))
        
        avg_images = []
        for name, df in dfs.items():
            text_image_counts = df.groupby('text')['path'].count()
            avg_images.append({
                'Dataset': name,
                'Avg Images': text_image_counts.mean()
            })
            print(f"\n{name} image statistics:")
            print(f"Average images per news: {text_image_counts.mean():.2f}")
            print(f"Max images: {text_image_counts.max()}")
            
        df_plot = pd.DataFrame(avg_images)
        sns.barplot(x='Dataset', y='Avg Images', data=df_plot)
        plt.title('Average Number of Images per News')
        plt.savefig('assets/image_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # # 设置中文字体
    # setup_chinese_font()
    
    # # 创建assets目录（如果不存在）
    # if not os.path.exists('assets'):
    #     os.makedirs('assets')
        
    # # 初始化分析器
    # analyzer = DatasetAnalyzer('./data')
    
    # # 加载数据集
    # datasets = analyzer.load_datasets()
    
    # # 进行各项分析
    # analyzer.print_statistics(datasets)
    # analyzer.analyze_distribution(datasets)
    # analyzer.analyze_text_length(datasets)
    # analyzer.analyze_image_distribution(datasets)
    analyze_dataset_statistics()
    print("\nAnalysis completed! Visualization results have been saved in the assets directory.")

if __name__ == "__main__":
    main()