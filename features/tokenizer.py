# -*- coding: utf-8 -*-


import os
import re

import jieba

import config


def load_set(path):
    lines = set()
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                lines.add(line)
    return lines


def tokenizer(line, filter_stopwords=False, cut_type='word'):
    line = str(line)
    if cut_type == 'word':
        seg_list = jieba.lcut(line)
    elif cut_type == 'char':
        seg_list = list(line)
    else:
        raise ValueError("cut type is word or char.")
    if filter_stopwords:
        stopwords = load_set(config.stopwords_path)
        seg_list = [x for x in seg_list if x not in stopwords]
    return seg_list


def sent_tokenizer(text, symbol_path=config.sentence_symbol_path):
    symbols = load_set(symbol_path)
    sentence_symbols = "".join(symbols)
    re_s = "[" + sentence_symbols + "]"
    short_sents = re.split(re_s, text)
    return short_sents
