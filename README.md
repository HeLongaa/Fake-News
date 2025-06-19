# 假新闻检测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

基于机器学习和自然语言处理的中文假新闻自动识别系统。

## 项目简介

本项目旨在通过分析文本内容的语言特征，自动识别和分类真实新闻（标签0）与虚假新闻（标签1）。项目综合运用了多种特征提取方法和模型算法，实现了高效准确的假新闻检测。

## 项目结构

```
Fake-News/
├── config.py                # 配置文件
├── preprocess.py            # 数据预处理
├── generate_features.py     # 特征生成
├── train_models.py          # 模型训练和预测
├── requirements.txt         # 依赖库列表
├── data/                    # 数据文件
│   ├── train_sample.csv     # 训练集样例
│   ├── test_sample.csv      # 测试集样例
│   ├── debunking_sample.csv # 辟谣数据样例
│   ├── stopwords.txt        # 停用词表
│   └── sentence_symbol.txt  # 句子分隔符
├── docs/                    # 文档和可视化
│   ├── data_virtual.ipynb   # 数据分析和可视化
│   ├── model.png            # 模型结构图
│   ├── stacking.png         # 模型集成图
│   └── train_data.png       # 训练数据分析图
├── features/                # 特征提取
│   ├── __init__.py
│   ├── char_tfidf_feature.py # 字符级TF-IDF
│   ├── count_feature.py      # 词频统计
│   ├── io_util.py            # IO工具
│   ├── math_util.py          # 数学工具
│   ├── ngram.py              # N-gram处理
│   ├── onehot_feature.py     # 独热编码
│   ├── sentiment_feature.py  # 情感分析
│   ├── svd_feature.py        # SVD降维
│   ├── tfidf_feature.py      # 词级TF-IDF
│   ├── tokenizer.py          # 文本分词
│   └── word2vec_feature.py   # 词向量特征
├── models/                  # 模型实现
│   ├── __init__.py
│   ├── base_model.py         # 模型基类
│   ├── bert_model.py         # BERT模型
│   ├── bert_tokenization.py  # BERT分词
│   ├── catboost_model.py     # CatBoost模型
│   ├── dpcnn_model.py        # DPCNN模型
│   ├── lr_model.py           # 逻辑回归模型
│   ├── rnn_model.py          # RNN模型
│   ├── score.py              # 评分工具
│   ├── textcnn_model.py      # TextCNN模型
│   └── xgboost_model.py      # XGBoost模型
└── submits/                 # 预测结果
    ├── catboost_submit.csv   # CatBoost提交结果
    ├── lr_submit.csv         # 逻辑回归提交结果
    ├── README.md             # 提交说明
    └── xgboost_submit.csv    # XGBoost提交结果
```

## 数据处理流程

1. **数据预处理**：
   - 加载训练集、测试集和辟谣数据集
   - 去除文本重复、过长和过短的样本
   - 合并数据集并保存为pickle格式

2. **特征提取**：
   - 使用jieba进行中文分词
   - 生成unigram、bigram和trigram特征
   - 应用多种特征提取器：
     * 词频统计
     * 字符级TF-IDF
     * 词级TF-IDF
     * SVD降维
     * Word2Vec词向量
     * 情感分析
     * 独热编码

3. **模型训练与预测**：
   - 传统机器学习模型：逻辑回归、XGBoost、CatBoost
   - 深度学习模型：TextCNN、RNN、DPCNN、BERT
   - 支持K折交叉验证和模型评估

## 主要模型

### 传统机器学习模型

- **逻辑回归**：基础线性分类器，计算速度快，可解释性好
- **XGBoost**：梯度提升树模型，处理非线性关系能力强
- **CatBoost**：针对类别特征优化的梯度提升树模型

### 深度学习模型

- **TextCNN**：使用卷积神经网络提取文本局部特征
- **RNN**：使用循环神经网络捕捉文本序列特征
- **DPCNN**：深度金字塔CNN，层次化提取文本特征
- **BERT**：预训练语言模型，通过微调适应假新闻检测任务

## 使用方法

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 数据预处理

```bash
python preprocess.py
```

### 特征生成

```bash
python generate_features.py
```

### 模型训练与预测

```bash
python train_models.py
```

默认只运行基础模型（逻辑回归），如需运行其他模型，请修改`train_models.py`中的主函数：

```python
if __name__ == '__main__':
    base_model()               # 运行基础模型
    # train_classic_models()   # 运行传统机器学习模型
    # train_deep_models()      # 运行深度学习模型
    # train_bert_model()       # 运行BERT模型
```

## 评估指标

- 准确率（Accuracy）
- 分类报告（Classification Report）
- 混淆矩阵（Confusion Matrix）

## 依赖库

- numpy
- pandas
- jieba
- scipy
- scikit-learn
- wordcloud
- pysenti
- xgboost
- catboost
- keras
- kashgari-tf

## 项目特点

1. **模块化设计**：特征提取器和模型采用面向对象设计，易于扩展和维护
2. **多特征融合**：结合文本统计特征、语义特征和情感特征
3. **多模型集成**：支持多种类型的模型训练和预测
4. **灵活配置**：通过配置文件集中管理参数
5. **完整流程**：包含从数据预处理到模型评估的全流程实现

## 数据可视化

项目提供了数据分析和可视化Jupyter Notebook（`docs/data_virtual.ipynb`），用于：
- 文本长度分布分析
- 标签分布分析
- 词云可视化
- 特征重要性分析

## 注意事项

- 首次运行需要处理大量文本数据，可能耗时较长
- 深度学习模型需要较高的计算资源
- BERT模型需要预先下载预训练模型，请修改`config.py`中的`pretrained_bert_path`

## 未来改进

- 添加更多文本特征，如情感极性、主题分布等
- 优化模型参数，提高检测准确率
- 增加模型可解释性分析
- 支持增量学习和在线预测
- 构建Web演示界面
