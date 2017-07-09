# -*- coding: utf-8 -*-
"""
@author: Fangshu Gao
@brief: Features in ChenglongChen's paper 4.1.6 - 4.1.10
@time: 2017/7/9 15:35 PM
"""
from utils import ngram_utils, dist_utils, np_utils
import config

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

# IntersectCount Ngram: count of word ngram of obs that closely matches any word ngram of target.
# How many ngrams of obs are in target?
# ob: [A, B, C，D] e.g., ["I", "am", "Denny"] 比如产品标题
# target: [A, B, C, E] 比如产品描述
# obs_ngrams(n=2): [AB, BC, CD]
# target_ngrams(n=2): [AB, BC, CE]
# ->
# IntersectCount: 4 (i.e., AB, AB, AB, AC)
# IntersectRatio: 4/6
def intersect_count_ngram(obs, target, n, join_string = ' ', not_need_ngram = 1):
    # n 的选择范围参照上面的 _ngram_str_map
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    obs_ngrams = ngram_utils._ngrams(obs, n, join_string)
    target_ngrams = ngram_utils._ngrams(target, n, join_string)
    count = 0.
    for w1 in obs_ngrams:
        for w2 in target_ngrams:
            if dist_utils._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):    # threshold=1.0
                count += 1.
                break
                # 如果 w1 == w2，则记 1 次, 注意 break
                # 对于 obs_ngrams(n=2): [AB, BC, CD] 和 target_ngrams(n=2): [AB, BC, CE] 则为 1+1=2
    if not_need_ngram:
        return count
    else:
        return [count, obs_ngrams]

# IntersectRatio Ngram: same as IntersectCount Ngram but normalized with the total number of word ngram in obs
def intersect_ratio_ngram(obs, target, n, join_string = ' '):
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    count_result = intersect_count_ngram(obs, target, n, join_string, not_need_ngram=0)
    count = count_result[0]
    return np_utils._try_divide(count, len(count_result[1]))

# CooccurrenceCount Ngram: count of closely matching word ngram pairs between obs and target.
# How many cooccurrence ngrams between obs and target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
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

# CooccurrenceRatio Ngram: same as CooccurrenceCount Ngram but normalized with the total number of word ngram pairs between obs and target.
def cooccurrence_ratio_ngram(obs, target, n, join_string=' '):
    assert type(obs) == list
    assert type(target) == list
    assert type(n) == int
    count_result = cooccurrence_count_ngram(obs, target, n, join_string, not_need_ngram=0)
    count = count_result[0]
    return np_utils._try_divide(count, len(count_result[1]) * len(count_result[2]))

# ------- 4.1.7 Intersect Count Features -------

# ------- 4.1.8 Intersect Position Features -------

# ------- 4.1.9 Match Features -------

# ------- 4.1.10 Query Quality Features -------