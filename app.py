import streamlit as st
import torch
from PIL import Image
import os
from models import Mynet
from config import Config
import torch.nn.functional as F
from transformers import BertTokenizer
import torchvision.transforms as transforms
import glob

class FakeNewsDetector:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.available_models = self.get_available_models()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def get_available_models(self):
        """获取所有已保存的模型文件"""
        model_files = glob.glob('model/*.ckpt')
        models = {}
        
        for model_file in model_files:
            filename = os.path.basename(model_file)
            # 解析文件名以获取BERT和ResNet类型
            if filename.startswith('S_'):  # 对比学习模型
                parts = filename[2:-5].split('_')  # 移除'S_'前缀和'.ckpt'后缀
                bert_type = parts[0]
                resnet_type = parts[1]
                
                # 创建用户友好的显示名称
                display_name = self.get_model_display_name(bert_type, resnet_type)
                models[display_name] = {
                    'path': model_file,
                    'bert_type': bert_type,
                    'resnet_type': resnet_type,
                    'use_sloss': True
                }
            else:
                parts = filename[:-5].split('_')  # 移除'.ckpt'后缀
                bert_type = parts[0]
                resnet_type = parts[1]
                
                display_name = self.get_model_display_name(bert_type, resnet_type)
                models[display_name] = {
                    'path': model_file,
                    'bert_type': bert_type,
                    'resnet_type': resnet_type,
                    'use_sloss': False
                }
        
        return models
    
    def get_model_display_name(self, bert_type, resnet_type):
        """生成用户友好的模型显示名称"""
        bert_names = {
            'bert-base-chinese': 'BERT-base中文',
            'chinese-bert-wwm-ext': 'BERT-wwm-ext中文',
            'minirbt-h256': 'MiniRBT-H256'
        }
        
        bert_display = bert_names.get(bert_type, bert_type)
        return f"{bert_display} + {resnet_type.upper()}"

    def load_model(self, model_info):
        """加载选定的模型"""
        self.config.bert_name = f'bert_model/{model_info["bert_type"]}'
        self.config.resnet_name = model_info['resnet_type']
        self.config.usesloss = model_info['use_sloss']
        
        # 更新模型配置
        if model_info['bert_type'] == 'minirbt-h256':
            self.config.bert_fc = 256
        else:
            self.config.bert_fc = 768
        self.config.resnet_fc = self.config.bert_fc
        
        # 加载模型
        model = Mynet(self.config).to(self.device)
        model.load_state_dict(torch.load(model_info['path'], map_location=self.device))
        model.eval()
        return model, BertTokenizer.from_pretrained(f'bert_model/{model_info["bert_type"]}')

    def predict(self, text, image, model, tokenizer):
        """进行预测"""
        # 处理文本
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.pad_size,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
        # 处理图像
        if image is not None:
            image = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # 如果没有图像，创建一个全零的图像张量
            image = torch.zeros((1, 3, 224, 224)).to(self.device)
        
        # 获取预测结果
        with torch.no_grad():
            text_input = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # 将输入组合成元组传递给模型
            inputs = (image, text_input, attention_mask)
            outputs = model(inputs)
            
            probabilities = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            if not isinstance(probabilities, torch.Tensor):
                probabilities = torch.tensor(probabilities)
            
            # 如果输出没有经过softmax，则应用softmax
            if probabilities.dim() == 2:  # logits
                probabilities = F.softmax(probabilities, dim=1)
                
        return probabilities.cpu().numpy()[0]

def main():
    st.set_page_config(
        page_title="多模态假新闻检测系统",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("多模态假新闻检测系统")
    
    detector = FakeNewsDetector()
    
    if not detector.available_models:
        st.error("未找到任何可用的模型文件！请确保模型文件存放在正确的位置。")
        return
    
    # 模型选择
    st.sidebar.header("模型配置")
    model_name = st.sidebar.selectbox(
        "选择模型",
        list(detector.available_models.keys())
    )
    
    # 显示模型信息
    model_info = detector.available_models[model_name]
    st.sidebar.write("模型信息：")
    st.sidebar.write(f"- BERT类型：{model_info['bert_type']}")
    st.sidebar.write(f"- ResNet类型：{model_info['resnet_type']}")
    st.sidebar.write(f"- 使用对比学习：{'是' if model_info['use_sloss'] else '否'}")
    
    # 主界面
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("文本输入")
        news_text = st.text_area("请输入新闻文本", height=200)
        
    with col2:
        st.subheader("图片上传")
        uploaded_file = st.file_uploader("上传新闻图片", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_column_width=True)
        
    # 检测按钮
    if st.button("开始检测", type="primary"):
        if not news_text:
            st.error("请输入新闻文本！")
            return
            
        try:
            # 加载选定的模型
            model, tokenizer = detector.load_model(model_info)
            
            # 处理图片
            image_data = Image.open(uploaded_file) if uploaded_file else None
            
            with st.spinner('正在进行检测...'):
                # 进行预测
                probabilities = detector.predict(news_text, image_data, model, tokenizer)
            
            # 显示结果
            st.subheader("检测结果")
            
            # 使用进度条显示概率
            col1, col2 = st.columns(2)
            with col1:
                st.write("真实新闻概率")
                st.progress(float(probabilities[0]))
                st.write(f"{probabilities[0]*100:.2f}%")
                
            with col2:
                st.write("虚假新闻概率")
                st.progress(float(probabilities[1]))
                st.write(f"{probabilities[1]*100:.2f}%")
            
            # 结论
            if probabilities[1] > probabilities[0]:
                st.error("⚠️ 系统判定：该新闻可能是虚假新闻！")
                if probabilities[1] > 0.8:
                    st.write("置信度：高")
                else:
                    st.write("置信度：中等")
            else:
                st.success("✅ 系统判定：该新闻可能是真实新闻。")
                if probabilities[0] > 0.8:
                    st.write("置信度：高")
                else:
                    st.write("置信度：中等")
                
        except Exception as e:
            st.error(f"检测过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main()