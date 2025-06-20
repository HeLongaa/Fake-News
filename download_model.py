from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.models as models
import os

def download_bert_model(model_name, save_dir):
    """
    下载BERT预训练模型并保存到指定路径
    Args:
        model_name (str): BERT模型名称，如 'bert-base-chinese', 'hfl/chinese-bert-wwm-ext', 'hfl/minirbt-h256'
        save_dir (str): 模型保存路径
    """

    print(f"开始下载 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 保存模型
    save_path = os.path.join(save_dir, model_name.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"{model_name} 下载完成！")

if __name__ == '__main__':
    # 下载所有模型
    bert_model = ['bert-base-chinese', 'hfl/chinese-bert-wwm-ext', 'hfl/minirbt-h256']
    
    bert_save_path = "./bert_model"

    for model_name in bert_model:
        download_bert_model(model_name, bert_save_path)