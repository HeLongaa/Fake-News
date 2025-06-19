# -*- coding: utf-8 -*-


import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
origin_train_file = pwd_path + '/data/train.csv'
origin_test_file = pwd_path + '/data/test_stage1.csv'
origin_debunking_file = pwd_path + '/data/debunking.csv'
stopwords_path = pwd_path + "/data/stopwords.txt"
sentence_symbol_path = pwd_path + '/data/sentence_symbol.txt'
output_dir = pwd_path + '/output/'

data_file_path = output_dir + 'data.pkl'
ngram_feature_path = output_dir + 'text_ngram.pkl'
vocab_path = output_dir + 'vocab.txt'

pretrained_bert_path = '/bert/chinese_L-12_H-768_A-12'
is_debug = False

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
