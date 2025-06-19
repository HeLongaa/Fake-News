# 中文虚假新闻检测系统

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
│   ├── 1_数据探索.ipynb
│   ├── 2_特征工程.ipynb
│   └── 3_模型训练与评估.ipynb
│
├── src/                       # 源代码
│   ├── preprocessing/         # 数据预处理模块
│   ├── features/              # 特征工程模块
│   ├── models/                # 模型实现
│   └── utils/                 # 工具函数
│
├── tests/                     # 测试代码
│
├── results/                   # 实验结果
│   ├── models/                # 保存的模型
│   └── plots/                 # 可视化图表
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
python src/train.py --model bert --epochs 5
```

3. **模型评估**

```bash
python src/evaluate.py --model_path results/models/bert_model.pkl
```

## 实验结果

主要模型性能比较（在测试集上）：

| 模型          | 准确率  | 精确率  | 召回率  | F1分数  |
|--------------|--------|--------|--------|--------|
| TF-IDF + SVM | 0.82   | 0.81   | 0.83   | 0.82   |
| BERT         | 0.91   | 0.90   | 0.92   | 0.91   |
| BiLSTM+Attn  | 0.87   | 0.86   | 0.88   | 0.87   |
| 融合模型      | 0.93   | 0.92   | 0.94   | 0.93   |

## 进一步改进

- 引入情感分析特征
- 构建用户信誉模型
- 增加更多社交网络传播特征
- 实现模型的在线更新

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

如有任何问题，请联系：your.email@example.com
