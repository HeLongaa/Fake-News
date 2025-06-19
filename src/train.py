import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import random_split
from transformers import BertTokenizer

from utils.data_loader import DataLoader
from utils.data_utils import RumorDataset, create_data_loaders
from models.neural_models import BERTClassifier
from utils.trainer import ModelTrainer
from preprocessing.text_processor import TextProcessor

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train rumor detection model')
    parser.add_argument('--model', type=str, default='bert', help='Model type (bert/bilstm)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results/models', help='Output directory')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    setup_logging()
    logging.info(f'Starting training with args: {args}')
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    data_loader = DataLoader()
    rumors_df = data_loader.load_rumors()
    original_df, rumor_reposts_df, non_rumor_reposts_df = data_loader.load_ced_dataset()
    
    # 预处理文本
    text_processor = TextProcessor()
    processed_texts = original_df['text'].apply(text_processor.clean_text).tolist()
    labels = original_df['label'].tolist()
    
    # 初始化tokenizer和数据集
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = RumorDataset(processed_texts, labels, tokenizer, args.max_len)
    
    # 划分数据集
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, args.batch_size
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'bert':
        model = BERTClassifier()
    else:
        raise ValueError(f'Unsupported model type: {args.model}')
    
    # 初始化训练器
    trainer = ModelTrainer(model, device, args.lr)
    
    # 训练模型
    logging.info('Starting training...')
    history = trainer.train(train_loader, val_loader, args.epochs)
    
    # 评估模型
    logging.info('Evaluating on test set...')
    test_metrics = trainer.evaluate(test_loader)
    logging.info(f'Test metrics: {test_metrics}')
    
    # 保存训练历史
    torch.save({
        'args': vars(args),
        'history': history,
        'test_metrics': test_metrics
    }, output_dir / 'training_history.pt')
    
    logging.info('Training completed!')

if __name__ == '__main__':
    main()
