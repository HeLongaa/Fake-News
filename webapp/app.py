import streamlit as st
import torch
import numpy as np
from pathlib import Path
import sys
import os
import json
import pandas as pd
from io import StringIO

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.text_processor import TextProcessor
from src.features.feature_extractor import FeatureExtractor
from src.models.neural_models import BERTClassifier, BiLSTMClassifier
from src.models.traditional_models import SVMClassifier, RandomForestClassifier
from transformers import BertTokenizer
import joblib
from src.utils.data_utils import extract_text_features

# 页面配置
st.set_page_config(
    page_title="虚假新闻检测系统",
    page_icon="🔍",
    layout="wide"
)

MODEL_PATHS = {
    'BERT': 'results/models/bert_model.pt',
    'BiLSTM': 'results/models/bilstm_model.pt',
    'SVM': 'results/models/svm_model.joblib',
    'RandomForest': 'results/models/rf_model.joblib'
}

@st.cache_resource
def load_models():
    """加载所有预训练模型"""
    models = {}
    
    # 加载神经网络模型
    if os.path.exists(MODEL_PATHS['BERT']):
        bert_model = BERTClassifier()
        bert_model.load_state_dict(torch.load(MODEL_PATHS['BERT'], map_location=torch.device('cpu')))
        bert_model.eval()
        models['BERT'] = bert_model

    if os.path.exists(MODEL_PATHS['BiLSTM']):
        bilstm_model = BiLSTMClassifier()
        bilstm_model.load_state_dict(torch.load(MODEL_PATHS['BiLSTM'], map_location=torch.device('cpu')))
        bilstm_model.eval()
        models['BiLSTM'] = bilstm_model
    
    # 加载传统机器学习模型
    if os.path.exists(MODEL_PATHS['SVM']):
        models['SVM'] = joblib.load(MODEL_PATHS['SVM'])
    
    if os.path.exists(MODEL_PATHS['RandomForest']):
        models['RandomForest'] = joblib.load(MODEL_PATHS['RandomForest'])
    
    # 加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 加载文本处理器和特征提取器
    text_processor = TextProcessor()
    feature_extractor = FeatureExtractor()
    
    return models, tokenizer, text_processor, feature_extractor

def predict_text(text, model_name, models, tokenizer, text_processor, feature_extractor):
    """使用选定的模型对输入文本进行预测"""
    # 文本预处理
    cleaned_text = text_processor.clean_text(text)
    
    # 根据模型类型选择不同的处理方式
    if model_name in ['BERT', 'BiLSTM']:
        # 神经网络模型处理
        if model_name == 'BERT':
            # BERT模型使用tokenizer
            encoded = tokenizer.encode_plus(
                cleaned_text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = models[model_name](
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
        else:
            # BiLSTM模型处理
            text_tensor = feature_extractor.text_to_tensor(cleaned_text)
            with torch.no_grad():
                outputs = models[model_name](text_tensor)
        
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    else:
        # 传统机器学习模型处理
        features = feature_extractor.extract_features(cleaned_text)
        features = np.array(features).reshape(1, -1)
        
        pred_class = models[model_name].predict(features)[0]
        proba = models[model_name].predict_proba(features)
        confidence = proba[0][pred_class]
    
    return pred_class, confidence, cleaned_text

def analyze_batch_texts(texts, model_name, models, tokenizer, text_processor, feature_extractor):
    """批量分析文本"""
    results = []
    for text in texts:
        pred_class, confidence, cleaned_text = predict_text(
            text, model_name, models, tokenizer, text_processor, feature_extractor
        )
        results.append({
            '原文': text,
            '预测结果': '虚假信息' if pred_class == 1 else '真实信息',
            '置信度': f"{confidence*100:.2f}%",
            '使用模型': model_name,
            '处理后文本': cleaned_text
        })
    return pd.DataFrame(results)

def main():
    # 加载所有模型和处理器
    models, tokenizer, text_processor, feature_extractor = load_models()
    
    if not models:
        st.error("未能加载任何模型，请确保模型文件存在！")
        return
    
    # 页面标题
    st.title("🔍 虚假新闻检测系统")
    
    # 选择模型
    available_models = list(models.keys())
    model_name = st.selectbox(
        "选择检测模型",
        available_models,
        help="BERT/BiLSTM：深度学习模型，适合复杂文本\nSVM/随机森林：传统机器学习模型，处理速度更快"
    )
    
    # 创建选项卡
    tab1, tab2 = st.tabs(["单条检测", "批量检测"])
    
    # 侧边栏
    st.sidebar.title("关于")
    st.sidebar.info(
        "这是一个基于深度学习的虚假新闻检测系统。"
        "模型使用大规模中文谣言数据集训练，"
        "可以帮助识别潜在的虚假信息。"
    )
    
    st.sidebar.title("使用说明")
    st.sidebar.markdown(
        """
        1. 在输入框中输入待检测的文本
        2. 点击"开始检测"按钮
        3. 系统会显示检测结果和置信度
        
        注意：
        - 输入文本长度不要超过512个字符
        - 结果仅供参考，建议结合其他信息源验证
        """
    )
    
    # 单条检测页面
    with tab1:
        st.markdown("### 单条文本检测")
        text_input = st.text_area(
            "请输入要检测的文本内容：",
            height=150,
            placeholder="在此输入新闻文本..."
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            detect_button = st.button("开始检测", use_container_width=True)
        with col2:
            st.markdown("")  # 占位
            show_detail = st.checkbox("显示详细分析")
        
        if detect_button:
            if not text_input:
                st.warning("请输入文本内容！")
            else:
                with st.spinner("正在分析..."):
                    # 进行预测
                    pred_class, confidence, cleaned_text = predict_text(
                        text_input, model_name, models, tokenizer, text_processor, feature_extractor
                    )
                
                # 显示结果
                st.markdown("## 检测结果")
                
                # 创建两列布局
                col1, col2 = st.columns(2)
                
                with col1:
                    if pred_class == 1:
                        st.error("⚠️ 可能是虚假信息")
                    else:
                        st.success("✅ 可能是真实信息")
                
                with col2:
                    st.metric(
                        label="置信度",
                        value=f"{confidence*100:.2f}%"
                    )
                
                    # 显示文本分析
                    if show_detail:
                        st.markdown("### 文本分析详情")
                        
                        # 使用expander显示详细信息
                        with st.expander("预处理后的文本", expanded=True):
                            st.text(cleaned_text)
                        
                        # 提取并显示文本特征
                        features = extract_text_features(cleaned_text)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            with st.expander("文本统计", expanded=True):
                                st.markdown(f"- 文本长度：{len(text_input)}")
                                st.markdown(f"- 字符数：{len(cleaned_text)}")
                                st.markdown(f"- 标点符号数：{features['punctuation_count']}")
                                st.markdown(f"- 情感倾向：{features['sentiment_score']:.2f}")
                        
                        with col2:
                            with st.expander("关键词", expanded=True):
                                keywords = text_processor.extract_keywords(cleaned_text)
                                st.write(", ".join(keywords[:10]))
                    
                    # 显示提醒
                    st.info(
                        "注意：此结果仅供参考，建议：\n"
                        "1. 查证信息来源\n"
                        "2. 核实关键事实\n"
                        "3. 参考权威媒体报道"
                    )
    
    # 批量检测页面
    with tab2:
        st.markdown("### 批量文本检测")
        st.markdown("支持上传txt或csv文件进行批量检测。文件格式要求：")
        st.markdown("- txt文件：每行一条文本")
        st.markdown("- csv文件：包含'text'列")
        
        uploaded_file = st.file_uploader("选择文件", type=['txt', 'csv'])
        
        if uploaded_file is not None:
            try:
                # 读取上传的文件
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    texts = df['text'].tolist()
                else:
                    content = uploaded_file.getvalue().decode('utf-8')
                    texts = [line.strip() for line in content.split('\\n') if line.strip()]
                
                # 显示处理进度
                with st.spinner(f'正在处理 {len(texts)} 条文本...'):
                    results_df = analyze_batch_texts(texts, model_name, models, tokenizer, text_processor, feature_extractor)
                
                # 显示结果
                st.markdown("### 检测结果")
                st.dataframe(results_df, use_container_width=True)
                
                # 提供下载选项
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "下载结果",
                    csv,
                    "检测结果.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            except Exception as e:
                st.error(f"处理文件时出错：{str(e)}")
                st.markdown("请确保文件格式正确。")

if __name__ == "__main__":
    main()
