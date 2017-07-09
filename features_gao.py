# -*- coding: utf-8 -*-
"""
@author: Fangshu Gao
@brief: Features in ChenglongChen's paper 4.1.6 - 4.1.11
@time: 2017/7/9 15:35 PM
"""
from collections import defaultdict
import numpy as np

from utils import ngram_utils, dist_utils, np_utils
import config
import google_spelling_checker_dict


# _ngram_str_map = {
#     1: "Unigram",
#     2: "Bigram",
#     3: "Trigram",
#     4: "Fourgram",
#     5: "Fivegram",
#     12: "UBgram",
#     123: "UBTgram",
# }

# ------- 4.1.6 Group Relevance Features 实际没用到，暂时没写 -------


# ------- 4.1.7 Intersect Count Features -------

# 4.1.7.1
# IntersectCount Ngram: count of word ngram of obs that closely matches any word ngram of target.
# How many ngrams of obs are in target?
# obs_ngrams(n=2): [AB, AB, AB, AC, DE, CD] 比如产品标题
# target_ngrams(n=2): [AB, AC, AB, AD, ED] 比如产品描述
# ->
# IntersectCount: 4 (i.e., AB, AB, AB, AC)
# IntersectRatio: 4/6
def intersect_count_ngram(obs, target, n, join_string=' ', not_need_ngram=1):
    # n 的选择范围参照上面的 _ngram_str_map
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    count = 0.
    for w1 in obs_ngrams:
        for w2 in target_ngrams:
            if dist_utils._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):  # threshold=1.0
                count += 1.
                break
                # 如果 w1 == w2，则记 1 次, 注意 break
                # 对于 obs_ngrams(n=2): [AB, BC, CD] 和 target_ngrams(n=2): [AB, BC, CE] 则为 1+1=2
    if not_need_ngram:
        return count
    else:
        return [count, obs_ngrams]


# 4.1.7.2
# IntersectRatio Ngram: same as IntersectCount Ngram but normalized with the total number of word ngram in obs
def intersect_ratio_ngram(obs, target, n, join_string=' '):
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    count_result = intersect_count_ngram(obs, target, n, join_string, not_need_ngram=0)
    count = count_result[0]
    return np_utils._try_divide(count, len(count_result[1]))


# 4.1.7.3
# CooccurrenceCount Ngram: count of closely matching word ngram pairs between obs and target.
# How many cooccurrence ngrams between obs and target?
# obs_ngrams(n=2): [AB, AB, AB, AC, DE, CD]
# target_ngrams(n=2): [AB, AC, AB, AD, ED]
# ->
# CooccurrenceCount: 7 (i.e., AB x 2 + AB x 2 + AB x 2 + AC x 1)
# CooccurrenceRatio: 7/(6 x 5)
def cooccurrence_count_ngram(obs, target, n, join_string=' ', not_need_ngram=1):
    # n 的选择范围参照上面的 _ngram_str_map
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    count = 0.
    for w1 in obs_ngrams:
        for w2 in target_ngrams:
            if dist_utils._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):
                count += 1.
    if not_need_ngram:
        return count
    else:
        return [count, obs_ngrams, target_ngrams]


# 4.1.7.4
# CooccurrenceRatio Ngram: same as CooccurrenceCount Ngram but normalized with the total number of word ngram pairs between obs and target.
def cooccurrence_ratio_ngram(obs, target, n, join_string=' '):
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    count_result = cooccurrence_count_ngram(obs, target, n, join_string, not_need_ngram=0)
    count = count_result[0]
    return np_utils._try_divide(count, len(count_result[1]) * len(count_result[2]))


# ------- 4.1.8 Intersect Position Features -------

def _inter_pos_list(obs, target):
    """
        Get the list of positions of obs in target
    """
    pos_list = [0]
    if len(obs) != 0:
        pos_list = [i for i, o in enumerate(obs, start=1) if o in target]
        if len(pos_list) == 0:
            pos_list = [0]
    return pos_list


def _inter_norm_pos_list(obs, target):
    pos_list = _inter_pos_list(obs, target)
    N = len(obs)
    return [np_utils._try_divide(i, N) for i in pos_list]


# 4.1.8.1
# IntersectPosition Ngram: see Sec 3.1.3 in Chenglong's Solution of Crowd-Flower for details
def intersect_position_ngram(obs, target, n, join_string=' '):
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    pos_list = _inter_pos_list(obs_ngrams, target_ngrams)
    return pos_list


# 4.1.8.2
# IntersectNormPosition Ngram: see Sec 3.1.3 in Chenglong's Solution of Crowd-Flower for details
def intersect_norm_position_ngram(obs, target, n, join_string=' '):
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    pos_list = _inter_norm_pos_list(obs_ngrams, target_ngrams)
    return pos_list


# ------- 4.1.9 Match Features -------

# 4.1.9.1
# MatchQueryCount: count of search term that occurs in target 即产品标题在产品描述中的数量
def match_query_count(obs, target, i_=0):
    def _str_whole_word(str1, str2, i_):
        cnt = 0
        if len(str1) > 0 and len(str2) > 0:
            try:
                while i_ < len(str2):
                    i_ = str2.find(str1, i_)
                    if i_ == -1:
                        return cnt
                    else:
                        cnt += 1
                        i_ += len(str1)
            except:
                pass
        return cnt

    return _str_whole_word(obs, target, i_)


# 4.1.9.2
# MatchQueryRatio: same as MatchQueryCount but normalized with the length of target
def match_query_ratio(obs, target, i_=0):
    return np_utils._try_divide(match_query_count(obs, target, i_), len(target.split(' ')))


# 4.1.9.3
# LongestMatchSize: longest match size between obs and target
def longest_match_size(obs, target):
    return dist_utils._longest_match_size(obs, target)


# 4.1.9.4
# LongestMatchRatio: same as LongestMatchSize but normalized with the minimum length of obs and target
def longest_match_ratio(obs, target):
    return dist_utils._longest_match_ratio(obs, target)


# 4.1.9.5
# MatchAttrCount: count of attribute in product attribute list that matches search term
def match_attr_count(obs, target, i_=0):
    def _str_whole_word(str1, str2, i_):
        cnt = 0
        if len(str1) > 0 and len(str2) > 0:
            try:
                while i_ < len(str2):
                    i_ = str2.find(str1, i_)
                    if i_ == -1:
                        return cnt
                    else:
                        cnt += 1
                        i_ += len(str1)
            except:
                pass
        return cnt

    cnt = 0
    for o in obs.split(" "):
        for t in target:
            if not t[0].startswith("bullet"):
                if _str_whole_word(obs, t[0], 0):
                    cnt += 1
    return cnt


# 4.1.9.6
# MatchAttrRatio: ratio of attribute in product attribute list that matches search term
def match_attr_ratio(obs, target, i_=0):
    lo = len(obs.split(" "))
    lt = len([t[0] for t in target if not t[0].startswith("bullet")])
    return np_utils._try_divide(match_attr_count(obs, target, i_), lo * lt)


# 4.1.9.7
# IsIndoorOutdoorMatch: whether the indoor/outdoor info in seach term and product attribute list match
# 我们的数据集不存在 indoor/outdoor 问题，暂时没写，但以后也许可以加上类似的分类问题


# ------- 4.1.10 Query Quality Features -------

# 4.1.10.1
# QueryQuality: measures the quality of search term using the edit distance among various versions, e.g., original search term and cleaned search term.
# 衡量产品名称到产品描述的修改难度(Levenshtein距离)
def query_quality(obs, target):
    return dist_utils._edit_dist(obs, target)


# 4.1.10.2
# IsInGoogleDict: whether the search term is in the Google spelling correction dictionary or not.
def IsInGoogleDict(obs, target):
    if obs in google_spelling_checker_dict.spelling_checker_dict:
        return 1.
    else:
        return 0.


# ------- 4.1.11 Statistical Cooccurrence TFIDF Features -------

# 4.1.11.1
# StatCoocTF Ngram: for each word ngram in obs, count how many word ngramin target that closely matches it and
# then aggregate to obtain various statistics,e.g., max and mean. StatCooc stands for StatisticalCooccurrence
def statcooc_tf_ngram(obs, target, n, join_string=' '):
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if dist_utils._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):
                s += 1.
        val_list.append(s)
    if len(val_list) == 0:
        val_list = [config.MISSING_VALUE_NUMERIC]
    return val_list


# 4.1.11.2
# StatCoocTFIDF Ngram: same as StatCoocTF Ngram but weight the count with IDF of the word ngram in the obs corpus
def statcooc_tfidf_ngram(obs, target, n, obs_corpus, target_corpus, join_string=' '):
    # TODO: obs_corpus, target_corpus 是什么？
    def _get_df_dict(target_corpus, n, join_string):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in target_corpus:
            target_ngrams = ngram_utils._ngrams(target, n, join_string)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(obs_corpus, word):
        N = len(obs_corpus)
        return np.log((N - _get_df_dict(target_corpus, n, join_string)[word] + 0.5) /
                      (_get_df_dict(target_corpus, n, join_string)[word] + 0.5))

    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if dist_utils._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):
                s += 1.
        val_list.append(s * _get_idf(obs_corpus, w1))
    if len(val_list) == 0:
        val_list = [config.MISSING_VALUE_NUMERIC]
    return val_list


# 4.1.11.3
# StatCoocBM25 Ngram: same as StatCoocTFIDF Ngram but with BM25 weighting
def StatCoocBM25_Ngram(obs, target, n, obs_corpus, target_corpus, join_string=' '):
    # TODO: obs_corpus, target_corpus 是什么？
    def _get_df_dict(target_corpus, n, join_string):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in target_corpus:
            target_ngrams = ngram_utils._ngrams(target, n, join_string)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(obs_corpus, word):
        N = len(obs_corpus)
        return np.log((N - _get_df_dict(target_corpus, n, join_string)[word] + 0.5) /
                      (_get_df_dict(target_corpus, n, join_string)[word] + 0.5))

    def _get_avg_ngram_doc_len():
        lst = []
        for target in target_corpus:
            target_ngrams = ngram_utils._ngrams(target, n, join_string)
            lst.append(len(target_ngrams))
        return np.mean(lst)

    k1 = config.BM25_K1  # TODO: k1, b 是什么？
    b = config.BM25_B
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    K = k1 * (1 - b + b * np_utils._try_divide(len(target_ngrams), _get_avg_ngram_doc_len()))
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if dist_utils._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):
                s += 1.
        bm25 = s * _get_idf(obs_corpus, w1) * np_utils._try_divide(1 + k1, s + K)
        val_list.append(bm25)
    if len(val_list) == 0:
        val_list = [config.MISSING_VALUE_NUMERIC]
    return val_list

# ------- 4.1.12 Vector Space Features -------




# ------- End -------
