import argparse
import torch
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

from models.neural_models import BERTClassifier
from utils.data_loader import DataLoader
from preprocessing.text_processor import TextProcessor
from transformers import BertTokenizer

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )

def evaluate_batch(model, texts, tokenizer, device, max_len=512):
    """评估一个批次的数据"""
    model.eval()
    with torch.no_grad():
        # 预处理文本
        encodings = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 将输入移到设备
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # 模型预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predictions = torch.max(outputs, dim=1)
        
        return predictions.cpu().numpy()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate rumor detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logging.info(f'Starting evaluation with args: {args}')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = BERTClassifier()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 加载数据
    data_loader = DataLoader()
    test_data = pd.read_json(args.data_path, lines=True)
    
    # 预处理文本
    text_processor = TextProcessor()
    processed_texts = test_data['text'].apply(text_processor.clean_text).tolist()
    
    # 批量预测
    predictions = []
    for i in tqdm(range(0, len(processed_texts), args.batch_size)):
        batch_texts = processed_texts[i:i + args.batch_size]
        batch_predictions = evaluate_batch(model, batch_texts, tokenizer, device)
        predictions.extend(batch_predictions)
    
    # 保存预测结果
    test_data['predicted_label'] = predictions
    test_data.to_csv(args.output_path, index=False)
    
    logging.info(f'Predictions saved to {args.output_path}')
    
    # 如果有真实标签，计算评估指标
    if 'label' in test_data.columns:
        from sklearn.metrics import classification_report
        report = classification_report(test_data['label'], predictions)
        logging.info(f'\nClassification Report:\n{report}')

if __name__ == '__main__':
    main()
