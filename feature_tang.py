# -*- coding: utf-8 -*-
"""
@author: Yifan Tang
@time: 2017/7/9
@brief: basic features

"""

import re
import config
import string

import scipy
import numpy as np

from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from utils import ngram_utils, nlp_utils, np_utils, dist_utils
# from utils import time_utils, logging_utils, pkl_utils

"""
1. Basic Features

"""

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


"""
2. Distance Features

"""

# 返回值是标题和描述之间的Jaccard距离


def JaccardCoef_Ngram(obs, target, ngram, token_pattern=" "):
    ngram_str = ngram_utils._ngram_str_map[ngram]
    print("JaccardCoef_%s" % ngram_str)
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    return dist_utils._jaccard_coef(obs_ngrams, target_ngrams)


# 返回值是标题和描述之间的Dice距离
def DiceDistance_Ngram(obs, target, ngram, token_pattern=" "):
    ngram_str = ngram_utils._ngram_str_map[ngram]
    print("JaccardCoef_%s" % ngram_str)
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    return dist_utils._dice_coef(obs_ngrams, target_ngrams)


# 返回值是标题和描述之间的edit距离(整体文档层面)
def EditDistance(obs, target):
    return dist_utils._edit_dist(obs, target)


# 返回值是标题和描述之间的edit距离(ngram层面)
def EditDistance_Ngram(obs, target, ngram, token_pattern=" ", aggregation_mode_prev="", aggregation_mode=""):
    ngram_str = ngram_utils._ngram_str_map[ngram]
    feat_name = []
    for m1 in aggregation_mode_prev:
        for m in aggregation_mode:
            n = "EditDistance_%s_%s_%s" % (
                ngram_str, string.capwords(m1), string.capwords(m))
            feat_name.append(n)
    print(feat_name)

    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    val_list = []
    for w1 in obs_ngrams:
        _val_list = []
        for w2 in target_ngrams:
            s = dist_utils._edit_dist(w1, w2)
            _val_list.append(s)
        if len(_val_list) == 0:
            _val_list = [config.MISSING_VALUE_NUMERIC]
        val_list.append(_val_list)
    if len(val_list) == 0:
        val_list = [[config.MISSING_VALUE_NUMERIC]]
    return val_list


# 返回值是标题和描述之间的compression距离(整体文档层面)
def CompressionDistance(obs, target):
    return dist_utils._compression_dist(obs, target)


# 返回值是标题和描述之间的compression距离(ngram层面)
def CompressionDistance_Ngram(obs, target, ngram, token_pattern=" ", aggregation_mode_prev="", aggregation_mode=""):
    ngram_str = ngram_utils._ngram_str_map[ngram]
    feat_name = []
    for m1 in aggregation_mode_prev:
        for m in aggregation_mode:
            n = "CompressionDistance_%s_%s_%s" % (
                ngram_str, string.capwords(m1), string.capwords(m))
            feat_name.append(n)
    print(feat_name)
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    val_list = []
    for w1 in obs_ngrams:
        _val_list = []
        for w2 in target_ngrams:
            s = dist_utils._compression_dist(w1, w2)
            _val_list.append(s)
        if len(_val_list) == 0:
            _val_list = [config.MISSING_VALUE_NUMERIC]
        val_list.append(_val_list)
    if len(val_list) == 0:
        val_list = [[config.MISSING_VALUE_NUMERIC]]
    return val_list


"""
3. Doc2Vec Features

"""

# 首先把原文件中Doc2Vec_BaseEstimator类中需要使用的函数拿出来定义


def Doc2Vec_get_vector(sent, doc2vec_model, sent_label):
    model = doc2vec_model
    sent_label = sent_label
    vector_size = doc2vec_model.vector_size
    try:
        vect = model.docvecs[sent_label[sent]]
    except:
        vect = np.zeros(vector_size, dtype=float)
    return vect


def Doc2Vec_get_vdiff(sent1, sent2):
    vect1 = Doc2Vec_get_vector(sent1)
    vect2 = Doc2Vec_get_vector(sent2)
    return dist_utils._vdiff(vect1, vect2)


def Doc2Vec_get_cosine_sim(sent1, sent2):
    vect1 = Doc2Vec_get_vector(sent1)
    vect2 = Doc2Vec_get_vector(sent2)
    return dist_utils._cosine_sim(vect1, vect2)


def Doc2Vec_get_rmse(sent1, sent2):
    vect1 = Doc2Vec_get_vector(sent1)
    vect2 = Doc2Vec_get_vector(sent2)
    return dist_utils._rmse(vect1, vect2)


# 返回值是对文本进行doc2vec后的结果(最后提交的模型中没有用这个feature)
def Doc2Vec_Vector(obs, doc2vec_model, model_prefix):
    vector_size = doc2vec_model.vector_size
    model_prefix = model_prefix
    print("Doc2Vec_%s_D%d_Vector" % (model_prefix, vector_size))
    return Doc2Vec_get_vector(obs)


# 返回值是对标题和描述进行doc2vec后两个结果的差
def Doc2Vec_Vdiff(obs, target, doc2vec_model, model_prefix):
    vector_size = doc2vec_model.vector_size
    model_prefix = model_prefix
    print("Doc2Vec_%s_D%d_Vdiff" % (model_prefix, vector_size))
    return Doc2Vec_get_vdiff(obs, target)


# 返回值是对标题和描述进行doc2vec后两个结果的余弦差
def Doc2Vec_CosineSim(obs, target, doc2vec_model, model_prefix):
    vector_size = doc2vec_model.vector_size
    model_prefix = model_prefix
    print("Doc2Vec_%s_D%d_CosineSim" % (model_prefix, vector_size))
    return Doc2Vec_get_cosine_sim(obs, target)


# 返回值是对标题和描述进行doc2vec后两个结果的RMSE
def Doc2Vec_RMSE(obs, target, doc2vec_model, model_prefix):
    vector_size = doc2vec_model.vector_size
    model_prefix = model_prefix
    print("Doc2Vec_%s_D%d_RMSE" % (model_prefix, vector_size))
    return Doc2Vec_get_rmse(obs, target)


"""
4. First and Last Ngram Features

"""

# 首先把原文件中Count_Ngram_BaseEstimator类中需要使用的函数拿出来定义


def Count_Ngram_get_match_count(obs, target, idx, str_match_threshold=config.STR_MATCH_THRESHOLD):
    str_match_threshold = str_match_threshold
    cnt = 0
    if (len(obs) != 0) and (len(target) != 0):
        for word in target:
            if dist_utils._is_str_match(word, obs[idx], str_match_threshold):
                cnt += 1
    return cnt


# idx = 0
def FirstIntersectCount_Ngram(obs, target, ngram, idx, str_match_threshold, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    return Count_Ngram_get_match_count(obs_ngrams, target_ngrams, idx)


# idx = -1
def LastIntersectCount_Ngram(obs, target, ngram, idx, str_match_threshold, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    return Count_Ngram_get_match_count(obs_ngrams, target_ngrams, idx)


# idx = 0
def FirstIntersectRatio_Ngram(obs, target, ngram, idx, str_match_threshold, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    return np_utils._try_divide(Count_Ngram_get_match_count(obs_ngrams, target_ngrams, idx), len(target_ngrams))


# idx = -1
def LastIntersectRatio_Ngram(obs, target, ngram, idx, str_match_threshold, token_pattern=" "):
    obs_tokens = nlp_utils._tokenize(obs, token_pattern)
    target_tokens = nlp_utils._tokenize(target, token_pattern)
    obs_ngrams = ngram_utils._ngrams(obs_tokens, ngram)
    target_ngrams = ngram_utils._ngrams(target_tokens, ngram)
    return np_utils._try_divide(Count_Ngram_get_match_count(obs_ngrams, target_ngrams, idx), len(target_ngrams))


"""
12. Vector Space Features

"""

# 首先把原文件中VectorSpace类中需要使用的函数拿出来定义


def VectorSpace_init_word_bow(ngram, vocabulary=None):
    bow = CountVectorizer(min_df=3,
                          max_df=0.75,
                          max_features=None,
                          # norm="l2",
                          strip_accents="unicode",
                          analyzer="word",
                          token_pattern=r"\w{1,}",
                          ngram_range=(1, ngram),
                          vocabulary=vocabulary)
    return bow


def VectorSpace_init_word_ngram_tfidf(ngram, vocabulary=None):
    tfidf = TfidfVectorizer(min_df=3,
                            max_df=0.75,
                            max_features=None,
                            norm="l2",
                            strip_accents="unicode",
                            analyzer="word",
                            token_pattern=r"\w{1,}",
                            ngram_range=(1, ngram),
                            use_idf=1,
                            smooth_idf=1,
                            sublinear_tf=1,
                            # stop_words="english",
                            vocabulary=vocabulary)
    return tfidf


def VectorSpace_init_char_tfidf(include_digit=False):
    chars = list(string.ascii_lowercase)
    if include_digit:
        chars += list(string.digits)
    vocabulary = dict(zip(chars, range(len(chars))))
    tfidf = TfidfVectorizer(strip_accents="unicode",
                            analyzer="char",
                            norm=None,
                            token_pattern=r"\w{1,}",
                            ngram_range=(1, 1),
                            use_idf=0,
                            vocabulary=vocabulary)
    return tfidf


def VectorSpace_init_char_ngram_tfidf(ngram, vocabulary=None):
    tfidf = TfidfVectorizer(min_df=3,
                            max_df=0.75,
                            max_features=None,
                            norm="l2",
                            strip_accents="unicode",
                            analyzer="char",
                            token_pattern=r"\w{1,}",
                            ngram_range=(1, ngram),
                            use_idf=1,
                            smooth_idf=1,
                            sublinear_tf=1,
                            # stop_words="english",
                            vocabulary=vocabulary)
    return tfidf


def VectorSpace_get_cooc_terms(lst1, lst2, join_str):
    out = [""] * len(lst1) * len(lst2)
    cnt = 0
    for item1 in lst1:
        for item2 in lst2:
            out[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(out)
    return res


def VectorSpace_CharDistribution_normalize(text):
    # pat = re.compile("[a-z0-9]")
    pat = re.compile("[a-z]")
    group = pat.findall(text.lower())
    if group is None:
        res = " "
    else:
        res = "".join(group)
        res += " "
    return res


def VectorSpace_CharDistribution_preprocess(corpus):
    return [VectorSpace_CharDistribution_normalize(text) for text in corpus]


def VectorSpace_CharDistribution_get_distribution(obs_corpus, target_corpus):
    # obs tfidf
    tfidf = VectorSpace_init_char_tfidf()
    X_obs = tfidf.fit_transform(
        VectorSpace_CharDistribution_preprocess(obs_corpus)).todense()
    X_obs = np.asarray(X_obs)
    # apply laplacian smoothing
    s = 1.
    X_obs = (X_obs + s) / (np.sum(X_obs, axis=1)[:, None] + X_obs.shape[1] * s)
    # targetument tfidf
    tfidf = VectorSpace_init_char_tfidf()
    X_target = tfidf.fit_transform(
        VectorSpace_CharDistribution_preprocess(target_corpus)).todense()
    X_target = np.asarray(X_target)
    X_target = (X_target + s) / (np.sum(X_target, axis=1)
                                 [:, None] + X_target.shape[1] * s)
    return X_obs, X_target


# 返回值是文本单词层面ngram的LSA
def LSA_Word_Ngram(obs_corpus, ngram, svd_dim=100, svd_n_iter=5):
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram)
    X = tfidf.fit_transform(obs_corpus)
    svd = TruncatedSVD(n_components=svd_dim,
                       n_iter=svd_n_iter, random_state=config.RANDOM_SEED)
    return svd.fit_transform(X)


# 返回值是文本字母层面ngram的LSA
def LSA_Char_Ngram(obs_corpus, ngram, svd_dim=100, svd_n_iter=5):
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram)
    X = tfidf.fit_transform(obs_corpus)
    svd = TruncatedSVD(n_components=svd_dim,
                       n_iter=svd_n_iter, random_state=config.RANDOM_SEED)
    return svd.fit_transform(X)


# 返回值是标题和描述中重复文本在单词层面ngram的LSA
def LSA_Word_Ngram_Cooc(obs_corpus, target_corpus, obs_ngram=1, target_ngram=1, svd_dim=100, svd_n_iter=5):
    obs_ngrams = list(map(lambda x: ngram_utils._ngrams(
        x.split(" "), obs_ngram, "_"), obs_corpus))
    target_ngrams = list(map(lambda x: ngram_utils._ngrams(
        x.split(" "), target_ngram, "_"), target_corpus))
    # cooccurrence ngrams
    cooc_terms = list(map(lambda lst1, lst2: VectorSpace_get_cooc_terms(
        lst1, lst2, "X"), obs_ngrams, target_ngrams))
    # tfidf
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram=1)
    X = tfidf.fit_transform(cooc_terms)
    # svd
    svd = TruncatedSVD(n_components=svd_dim,
                       n_iter=svd_n_iter, random_state=config.RANDOM_SEED)
    return svd.fit_transform(X)


# 返回值是标题的TFIDF和描述的TFIDF合并后在单词层面ngram的LSA
def LSA_Word_Ngram_Pair(obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram=ngram)
    X_obs = tfidf.fit_transform(obs_corpus)
    X_target = tfidf.fit_transform(target_corpus)
    X_tfidf = scipy.sparse.hstack([X_obs, X_target]).tocsr()
    # svd
    svd = TruncatedSVD(n_components=svd_dim,
                       n_iter=svd_n_iter, random_state=config.RANDOM_SEED)
    X_svd = svd.fit_transform(X_tfidf)
    return X_svd


# 返回值是标题单词层面LSA和描述单词层面LSA之间的余弦相似度
def LSA_Word_Ngram_CosineSim(obs_corpus, target_corpus, ngram=3, svd_dim=100, svd_n_iter=5):
    # get common vocabulary
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram)
    tfidf.fit(list(obs_corpus) + list(target_corpus))
    vocabulary = tfidf.vocabulary_
    # obs tfidf
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram, vocabulary)
    X_obs = tfidf.fit_transform(obs_corpus)
    # targetument tfidf
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram, vocabulary)
    X_target = tfidf.fit_transform(target_corpus)
    # svd
    svd = TruncatedSVD(n_components=svd_dim,
                       n_iter=svd_n_iter, random_state=config.RANDOM_SEED)
    svd.fit(scipy.sparse.vstack((X_obs, X_target)))
    X_obs = svd.transform(X_obs)
    X_target = svd.transform(X_target)
    # cosine similarity
    sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
    sim = np.asarray(sim).squeeze()
    return sim


# 返回值是标题字母层面LSA和描述字母层面LSA之间的余弦相似度
def LSA_Char_Ngram_CosineSim(obs_corpus, target_corpus, ngram=5, svd_dim=100, svd_n_iter=5):
    # get common vocabulary
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram)
    tfidf.fit(list(obs_corpus) + list(target_corpus))
    vocabulary = tfidf.vocabulary_
    # obs tfidf
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram, vocabulary)
    X_obs = tfidf.fit_transform(obs_corpus)
    # targetument tfidf
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram, vocabulary)
    X_target = tfidf.fit_transform(target_corpus)
    # svd
    svd = TruncatedSVD(n_components=svd_dim,
                       n_iter=svd_n_iter, random_state=config.RANDOM_SEED)
    svd.fit(scipy.sparse.vstack((X_obs, X_target)))
    X_obs = svd.transform(X_obs)
    X_target = svd.transform(X_target)
    # cosine similarity
    sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
    sim = np.asarray(sim).squeeze()
    return sim


# 返回值是标题单词层面的TFIDF和描述单词层面的TFIDF之间的余弦相似度
def TFIDF_Word_Ngram_CosineSim(obs_corpus, target_corpus, ngram=3):
    # get common vocabulary
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram)
    tfidf.fit(list(obs_corpus) + list(target_corpus))
    vocabulary = tfidf.vocabulary_
    # obs tfidf
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram, vocabulary)
    X_obs = tfidf.fit_transform(obs_corpus)
    # targetument tfidf
    tfidf = VectorSpace_init_word_ngram_tfidf(ngram, vocabulary)
    X_target = tfidf.fit_transform(target_corpus)
    # cosine similarity
    sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
    sim = np.asarray(sim).squeeze()
    return sim


# 返回值是标题字母层面的TFIDF和描述字母层面的TFIDF之间的余弦相似度
def TFIDF_Char_Ngram_CosineSim(obs_corpus, target_corpus, ngram=5):
    # get common vocabulary
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram)
    tfidf.fit(list(obs_corpus) + list(target_corpus))
    vocabulary = tfidf.vocabulary_
    # obs tfidf
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram, vocabulary)
    X_obs = tfidf.fit_transform(obs_corpus)
    # targetument tfidf
    tfidf = VectorSpace_init_char_ngram_tfidf(ngram, vocabulary)
    X_target = tfidf.fit_transform(target_corpus)
    # cosine similarity
    sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
    sim = np.asarray(sim).squeeze()
    return sim


# TSNE_LSA_Word_Ngram_Cooc没有了，之后做到这里的时候可以搞明白然后写一个


# 返回值是LSA_Word_Ngram_Pair()的二维TSNE
def TSNE_LSA_Word_Ngram_Pair(obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
    X_svd = LSA_Word_Ngram_Pair()
    X_scaled = StandardScaler().fit_transform(X_svd)
    X_tsne = TSNE().fit_transform(X_scaled)
    return X_tsne


# 返回值是标题字母分布和描述字母分布之间的比值
def CharDistribution_Ratio(obs_corpus, target_corpus, const_A=1., const_B=1.):
    X_obs, X_target = VectorSpace_CharDistribution_get_distribution()
    ratio = (X_obs + const_A) / (X_target + const_B)
    return ratio


# 返回值是标题字母分布和描述字母分布之间的余弦相似度
def CharDistribution_CosineSim(obs_corpus, target_corpus):
    X_obs, X_target = VectorSpace_CharDistribution_get_distribution()
    # cosine similarity
    sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
    sim = np.asarray(sim).squeeze()
    return sim


# 返回值是标题字母分布和描述字母分布之间的KL divergence
def CharDistribution_KL(obs_corpus, target_corpus):
    X_obs, X_target = VectorSpace_CharDistribution_get_distribution()
    # KL
    kl = dist_utils._KL(X_obs, X_target)
    return kl
