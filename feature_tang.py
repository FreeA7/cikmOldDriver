# -*- coding: utf-8 -*-
"""
@author: Yifan Tang
@time: 2017/7/9
@brief: basic features

"""
import numpy as np
import re

from collections import Counter
from utils import ngram_utils, nlp_utils, np_utils
from utils import time_utils, logging_utils, pkl_utils
from sklearn.preprocessing import LabelBinarizer


# 返回值为每一个观测值的标准形式数字编码
def DocId(obs):
    obs_set = set(obs)
    obs_encoder = dict(zip(obs_set, range(len(obs_set))))
    return obs_encoder[obs]


# 返回值为每一个观测值的OneHot形式数字编码
def DocIdOneHot(obs):
    lb = LabelBinarizer(sparse_output=True)
    return lb.fit_transform(obs)


# 返回值为一个包含3个国家哑变量的list
def NationDummy(obs):
    my_dummy = int(obs == 'my')
    sg_dummy = int(obs == 'sg')
    ph_dummy = int(obs == 'ph')
    return [my_dummy, sg_dummy, ph_dummy]


# 返回值为文本的长度(以单词计)，token_pattern默认为空格(" ")
def DocLen(obs, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    return len(obs_tokens)


# 返回值暂时没搞清楚，原文是: frequency of obs in the corpus
def DocFreq(obs):
    counter = Counter(obs)
    return counter[obs]


# 返回值为文本的熵
def DocEntropy(obs, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    counter = Counter(obs_tokens)
    count = np.asarray(list(counter.values()))
    proba = count / np.sum(count)
    return np_utils._entropy(proba)


# 返回值为文本中的标点数
def DigitCount(obs):
    return len(re.findall(r"\d", obs))


# 返回值为文本中标点占总字数的比例
def DigitRatio(obs, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    return np_utils._try_divide(len(re.findall(r"\d", obs)), len(obs_tokens))


# 返回值为文本中唯一的ngram个数
def UniqueCount_Ngram(obs, ngram, token_pattern=" "):
    ngram_str = ngram_utils._ngram_str_map[ngram]
    print("UniqueCount_%s" % ngram_str)
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    return len(set(obs_ngrams))


# 返回值为文本中唯一的ngram占总ngram的比例
def UniqueRatio_Ngram(obs, ngram, token_pattern=" "):
    ngram_str = ngram_utils._ngram_str_map[ngram]
    print("UniqueCount_%s" % ngram_str)
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    return np_utils._try_divide(len(set(obs_ngrams)), len(obs_ngrams))


# 返回值是介绍中特性的个数
def AttrCount(obs):
    """obs is a list of attributes"""
    return len(obs)


# 返回值是介绍中bullet point的个数
def AttrBulletCount(obs):
    """obs is a list of attributes"""
    cnt = 0
    for lst in obs:
        if lst[0].startswith("bullet"):
            cnt += 1
    return cnt


# 返回值是介绍中bullet point占总字数的比例
def AttrBulletRatio(obs):
    """obs is a list of attributes"""
    cnt = 0
    for lst in obs:
        if lst[0].startswith("bullet"):
            cnt += 1
    return np_utils._try_divide(cnt, len(obs))


# 返回值是介绍中非bullet point的个数
def AttrNonBulletCount(obs):
    """obs is a list of attributes"""
    cnt = 0
    for lst in obs:
        if not lst[0].startswith("bullet"):
            cnt += 1
    return cnt


# 返回值是介绍中非bullet point占总字数的比例
def AttrNonBulletRatio(obs):
    """obs is a list of attributes"""
    cnt = 0
    for lst in obs:
        if not lst[0].startswith("bullet"):
            cnt += 1
    return np_utils._try_divide(cnt, len(obs))


# 返回值是介绍中是否有高度的dummy
def AttrHasProductHeight(obs):
    for lst in obs:
        if lst[0].find("product height") != -1:
            return 1
    return 0


# 返回值是介绍中是否有宽度的dummy
def AttrHasProductWidth(obs):
    for lst in obs:
        if lst[0].find("product width") != -1:
            return 1
    return 0


# 返回值是介绍中是否有长度的dummy
def AttrHasProductLength(obs):
    for lst in obs:
        if lst[0].find("product length") != -1:
            return 1
    return 0


# 返回值是介绍中是否有深度的dummy
def AttrHasProductDepth(obs):
    for lst in obs:
        if lst[0].find("product depth") != -1:
            return 1
    return 0


# 返回值是介绍中是否有户外和室内的dummy
def AttrHasIndoorOutdoor(obs):
    for lst in obs:
        if lst[0].find("indoor outdoor") != -1:
            return 1
    return 0
