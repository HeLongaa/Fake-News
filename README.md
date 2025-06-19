# 中文虚假新闻检测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 项目概述

本项目基于中文社交媒体谣言数据集，构建了一个自动检测虚假新闻的机器学习系统。系统能够通过分析微博文本内容及其传播特征，有效识别潜在的虚假信息，为社交媒体平台的内容审核和用户提供有力的信息甄别支持。

## 数据集介绍

本项目使用来自新浪微博不实信息举报平台的中文谣言数据集，该数据集分为两个部分：

1. **原始谣言微博数据** (rumors_v170613.json)：
   - 包含2009年至2017年的31669条谣言原微博
   - 包含谣言编码、标题、举报者信息、谣言内容、审查结果等字段

2. **CED_Dataset**：
   - 包含1538条谣言和1849条非谣言
   - 不仅包含原微博内容，还包含对应的转发与评论信息
   - 结构分为：原始微博(original-microblog)、谣言转发(rumor-repost)和非谣言转发(non-rumor-repost)

## 技术方案

本项目采用以下步骤和技术实现虚假新闻检测：

1. **数据预处理**：
   - 文本清洗与规范化
   - 中文分词与停用词过滤
   - 特征抽取（文本特征、用户特征、传播特征）

2. **模型构建**：
   - 基础文本特征模型（TF-IDF、Word2Vec等）
   - 深度学习模型（BERT、BiLSTM等）
   - 融合模型（结合文本内容和传播特征）

3. **评估指标**：
   - 准确率(Accuracy)
   - 精确率(Precision)
   - 召回率(Recall)
   - F1分数
   - ROC曲线和AUC

## 项目结构

```
Fake-News/
│
├── data/                      # 数据目录
│   ├── rumors_v170613.json    # 原始谣言数据
│   └── CED_Dataset/           # 包含转发信息的谣言数据集
│       ├── original-microblog/ # 原始微博
│       ├── rumor-repost/      # 谣言转发
│       └── non-rumor-repost/  # 非谣言转发
│
├── notebooks/                 # Jupyter笔记本
│   ├── 1_数据探索.ipynb        # 数据集分析与可视化
│   ├── 2_特征工程.ipynb        # 特征提取与分析
│   └── 3_模型训练与评估.ipynb   # 模型实验与比较
│
├── src/                       # 源代码
│   ├── preprocessing/         # 数据预处理模块
│   │   └── text_processor.py  # 文本清洗与分词
│   ├── features/             # 特征工程模块
│   │   └── feature_extractor.py # 特征提取器
│   ├── models/               # 模型实现
│   │   ├── neural_models.py  # 深度学习模型（BERT、BiLSTM）
│   │   └── traditional_models.py # 传统机器学习模型（SVM、随机森林）
│   └── utils/                # 工具函数
│       ├── data_loader.py    # 数据加载
│       ├── data_utils.py     # 数据处理工具
│       └── trainer.py        # 模型训练器
│
├── webapp/                    # Web应用
│   └── app.py                # Streamlit应用入口
│
├── results/                   # 实验结果
│   ├── models/               # 保存的模型
│   │   ├── bert_model.pt     # BERT模型
│   │   ├── bilstm_model.pt   # BiLSTM模型
│   │   ├── svm_model.joblib  # SVM模型
│   │   └── rf_model.joblib   # 随机森林模型
│   └── plots/                # 可视化图表
│
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

## 使用说明

### 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/Fake-News.git
cd Fake-News

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

数据集已包含在项目中，但如需重新下载或更新，请参考`data/README.md`中的说明。

### 运行示例

1. **数据探索与预处理**
```bash
jupyter notebook notebooks/1_数据探索.ipynb
```

2. **模型训练**
```bash
# 训练BERT模型
python src/train.py --model bert --epochs 5

# 训练BiLSTM模型
python src/train.py --model bilstm --epochs 10

# 训练传统机器学习模型
python src/train.py --model svm
python src/train.py --model random_forest
```

3. **模型评估**
```bash
python src/evaluate.py --model bert --model_path results/models/bert_model.pt
```

4. **启动Web应用**
```bash
# 启动Streamlit应用
streamlit run webapp/app.py
```

Web应用提供以下功能：
- 支持多个检测模型的选择（BERT、BiLSTM、SVM、随机森林）
- 单条文本检测：输入文本直接检测
- 批量检测：支持上传txt或csv文件进行批量检测
- 详细分析：显示文本特征、关键词和置信度
- 检测结果导出：支持将批量检测结果导出为CSV文件
```

## 实验结果

主要模型性能比较（在测试集上）：

| 模型          | 准确率  | 精确率  | 召回率  | F1分数  | 推理速度 |
|--------------|--------|--------|--------|--------|---------|
| BERT         | 0.91   | 0.90   | 0.92   | 0.91   | 较慢    |
| BiLSTM       | 0.87   | 0.86   | 0.88   | 0.87   | 中等    |
| SVM          | 0.82   | 0.81   | 0.83   | 0.82   | 快     |
| 随机森林      | 0.84   | 0.83   | 0.85   | 0.84   | 快     |

各模型特点：
- **BERT**：性能最好，适合复杂文本的理解，但推理速度较慢
- **BiLSTM**：性能良好，平衡了准确性和速度
- **SVM**：速度快，适合实时检测，性能尚可
- **随机森林**：可解释性好，训练和推理速度快

## 进一步改进

- 模型优化：
  - 模型集成与投票机制
  - 引入跨语言预训练模型
  - 优化模型推理速度

- 特征增强：
  - 引入情感分析特征
  - 构建用户信誉模型
  - 增加传播网络特征

- 系统功能：
  - 支持更多文件格式的批量检测
  - 添加API接口支持
  - 实现模型的在线更新
  - 优化前端交互体验

## 引用

如果您使用了本项目，请引用以下论文：

```
@article{liu2015rumors,
  title={中文社交媒体谣言统计语义分析},
  author={刘知远 and 张乐 and 涂存超 and 孙茂松},
  journal={中国科学: 信息科学},
  volume={12},
  pages={1536--1546},
  year={2015}
}

@article{song2018ced,
  title={CED: Credible Early Detection of Social Media Rumors},
  author={Song, Changhe and Tu, Cunchao and Yang, Cheng and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:1811.04175},
  year={2018}
}
```

## 许可证

MIT License

## 联系方式

如有任何问题，请联系：helong_001@qq.com
