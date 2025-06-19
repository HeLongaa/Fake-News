# -*- coding: utf-8 -*-
"""
@description: 数据预处理
1. 连接训练数据和测试数据
2. 删除重复项并修剪长文本
"""

import pandas as pd

import config

label_dict = {'0': 0,  # true
              '1': 1  # fake
              }

# ----------------------load data--------------------------------

df_tr = pd.read_csv(config.origin_train_file)
df_tr['type'] = 'train'
print('df_tr.shape:', df_tr.shape)
# 文本去重
df_tr = df_tr.drop_duplicates(["text"], keep="first")
print('df_tr drop_duplicates:', df_tr.shape)

# 文本去太长的, 删除681个长的
df_tr = df_tr[df_tr['text'].str.len() < 300]
print('df_tr del long text:', df_tr.shape)
df_tr = df_tr[df_tr['text'].str.len() > 2]
print('df_tr del short text:', df_tr.shape)
print('df_tr head:\n', df_tr.head(2))
print('df_tr tail:\n', df_tr.tail(2))
print('df_tr value counts:\n', df_tr['label'].value_counts())

df_te = pd.read_csv(config.origin_test_file)
df_te['label'] = 0
df_te['type'] = 'test'
print('df_te.shape:', df_te.shape)
print('df_te head:\n', df_te.head(2))
# train count: 38471， del duplicates count: 33414, del long: 32733
# test count:  4000

# ---------------------- deal debunking data
df_ext = pd.read_csv(config.origin_debunking_file, usecols=['id', 'text'])
print('df_ext.shape:', df_ext.shape)
# 文本去重
df_ext = df_ext.drop_duplicates(["text"], keep="first")
# 补充到训练数据
df_ext['label'] = 1
df_ext['type'] = 'train'
print('df_ext text drop_duplicates:', df_ext.shape)
df_ext = df_ext.dropna(axis=0)
print('df_ext drop_na shape:', df_ext.shape)
print('df_ext head:\n', df_ext.head(2))
# 文本去太长的, 删除681个长的
df_ext = df_ext[df_ext['text'].str.len() < 300]
print('df_ext del long text:', df_ext.shape)
df_ext = df_ext[df_ext['text'].str.len() > 2]
print('df_ext del short text:', df_ext.shape)
print("df_ext text:", df_ext["text"][:5])

df_all = pd.concat([df_tr, df_ext, df_te], axis=0, sort=False)
print('df_all shape', df_all.shape)
print('df_all head:\n', df_all.head())
print('df_all tail:\n', df_all.tail())
df_all.to_pickle(config.data_file_path)
