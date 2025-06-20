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



def download_resnet_model(model_name,save_dir):
    """    
    下载ResNet预训练模型并保存到指定路径
    Args:
        model_name (str): ResNet模型名称，如 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        save_path (str): 模型保存路径
    """
    print(f"正在下载 {model_name} 预训练模型...")
    
    # 根据模型名称获取对应的模型
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    save_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"{model_name} 已保存到 {save_path}")



if __name__ == '__main__':
    # 下载所有模型
    resnet_model = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    bert_model = ['bert-base-chinese', 'hfl/chinese-bert-wwm-ext', 'hfl/minirbt-h256']
    
    resnet_save_path = "./pretrained_models"
    bert_save_path = "./bert_model"
    for model_name in resnet_model:
        download_resnet_model(model_name, resnet_save_path)

    for model_name in bert_model:
        download_bert_model(model_name, bert_save_path)