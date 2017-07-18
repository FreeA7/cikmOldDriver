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


# ------- Chenglong Chen's part ends & Then start Igor&Kostia’s part -------

#################################################################
### 4.2.3 Measure whether brand and materials in query match other text
#################################################################

### the following function compares query with the corresponding product title,
### prodcut description or attribute bullets and returns:
### * number of brands/materials fully matched
### * number of brands/materials partially matched ('behr' and 'behr premium')
### * number of brands/materials with assumed match (first word of brand found, e.g. 'handy home products' and 'handy paint pail')
### * number of brands/materials in query not matched
### * convoluted output: 3 if all brands/materials fully matched,
###                      2 if all matched at least partially
###                      1 if at least one is matched but at least one is not matched
###                      0 if no brand/material in query
###                     -1 if there is brand/material in query but no brand/material in text
###                     -2 if there are brands different brands/materials in query and text

def query_brand_material_in_attribute(str_query_brands, str_attribute_brands):
    list_query_brands = list(set(str_query_brands.split(";")))
    list_attribute_brands = list(set(str_attribute_brands.split(";")))
    while '' in list_query_brands:
        list_query_brands.remove('')
    while '' in list_attribute_brands:
        list_attribute_brands.remove('')

    str_attribute_brands = " ".join(str_attribute_brands.split(";"))
    full_match = 0
    partial_match = 0
    assumed_match = 0
    no_match = 0
    num_of_query_brands = len(list_query_brands)
    num_of_attribute_brands = len(list_attribute_brands)
    if num_of_query_brands > 0:
        for brand in list_query_brands:
            if brand in list_attribute_brands:
                full_match += 1
            elif ' ' + brand + ' ' in ' ' + str_attribute_brands + ' ':
                partial_match += 1
            elif (' ' + brand.split()[0] in ' ' + str_attribute_brands and brand.split()[0][0] not in "0123456789") or \
                    (len(brand.split()) > 1 and (
                                ' ' + brand.split()[0] + ' ' + brand.split()[1]) in ' ' + str_attribute_brands):
                assumed_match += 1
            else:
                no_match += 1

    convoluted_output = 0  # no brand in query
    if num_of_query_brands > 0:
        if num_of_attribute_brands == 0:
            convoluted_output = -1  # no brand in text, but there is brand in query
        elif no_match == 0:
            if assumed_match == 0:
                convoluted_output = 3  # all brands fully matched
            else:
                convoluted_output = 2  # all brands matched at least partially
        else:
            if full_match + partial_match + assumed_match > 0:
                convoluted_output = 1  # one brand matched but the other is not
            else:
                convoluted_output = -2  # brand mismatched

    return full_match, partial_match, assumed_match, no_match, convoluted_output

'''
#################################################################
### STEP 9: 'Query expansion'
#################################################################

### For each comination of beforethekey_thekey in query
### create the list of the most common words from product description.
### Since the average relevance tends to be closer to 3 than to 1,
### the majority of matched products are *relevant*. It means that
### the most common words in matched product description denote
### high level of relevance. So, we can assess relevance by estimating
### how many common words the prodcut description contains.

t2 = time()
df_all['search_term_beforethekey_thekey_stemmed'] = df_all['search_term_beforethekey_stemmed'] + "_" + df_all[
    'search_term_thekey_stemmed']
aa = list(set(list(df_all['search_term_beforethekey_thekey_stemmed'])))
similarity_dict = {}
for i in range(0, len(aa)):
    # get unique words from each product description then concatenate all results:
    all_descriptions = " ".join(list(
        df_all['product_description_stemmed_woBrand'][df_all['search_term_beforethekey_thekey_stemmed'] == aa[i]].map(
            lambda x: " ".join(list(set(x.split()))))))
    # and transform to a list:
    all_descriptions_list = all_descriptions.split()
    # vocabulary is simly a set of unique words:
    vocabulary = list(set(all_descriptions_list))
    # count the frequency of each combination of beforethekey_thekey
    cnt = list(df_all['search_term_beforethekey_thekey_stemmed']).count(aa[i])
    freqs = [1.0 * all_descriptions_list.count(w) / cnt for w in vocabulary]

    vocabulary += ['dummyword0', 'dummyword1', 'dummyword2', 'dummyword3', 'dummyword4', 'dummyword5', \
                   'dummyword6', 'dummyword7', 'dummyword8', 'dummyword9', 'dummyword10', 'dummyword11', 'dummyword12', \
                   'dummyword13', 'dummyword14', 'dummyword15', 'dummyword16', 'dummyword17', 'dummyword18', \
                   'dummyword19', 'dummyword20', 'dummyword21', 'dummyword22', 'dummyword23', 'dummyword24',
                   'dummyword25', \
                   'dummyword26', 'dummyword27', 'dummyword28', 'dummyword29']
    freqs += list(np.zeros(30))
    similarity_dict[aa[i]] = {"cnt": cnt,
                              'words': sorted(zip(vocabulary, freqs), key=lambda x: x[1], reverse=True)[0:30]}

    if (i % 2000) == 0:
        print "" + str(i) + " out of " + str(len(aa)) + " unique combinations; " + str(
            round((time() - t2) / 60, 1)) + " minutes"

print 'create similarity dict time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()

### check the next var for potential overfit
# df_all['frequency_of_beforethekey_thekey']=df_all['search_term_beforethekey_thekey_stemmed'].map(lambda x: int(similarity_dict[x]['cnt']) * int(x!="_"))

df_all['above8_dummy_frequency_of_beforethekey_thekey'] = df_all['search_term_beforethekey_thekey_stemmed'].map(
    lambda x: int(similarity_dict[x]['cnt'] >= 8 and x != "_"))


### Get the frequency of top words
### Assume the product descriptions consist only of three words:
### word1 appears in 80% of descriptions
### word2 appears in 60% of descriptions
### word3 appears in 40% of descriptions
### Then we encounter a product description with word1 and word3, but no word2.
### In this example the sum frequencies for matched words is 0.8+0.4=1.2;
### the sum of frequencies for all words is 0.8+0.6+0.4=1.8
def get_description_similarity(str_thekey_pair, str_description, similarity_dict_item):
    values = np.zeros(30)
    values0 = np.zeros(30)
    if str_thekey_pair != "_" and similarity_dict_item['cnt'] >= 8:
        for j in range(0, 30):
            ## frequencies for matched words
            values[j] = similarity_dict_item['words'][j][1] * int(
                " " + similarity_dict_item['words'][j][0] + " " in " " + str_description + " ")
            # frequencies for all words
            values0[j] = similarity_dict_item['words'][j][1]
    return sum(values[0:10]), sum(values[0:20]), sum(values[0:30]), sum(values0[0:10]), sum(values0[0:20]), sum(
        values0[0:30])


df_all['description_similarity_tuple'] = df_all.apply(lambda x: \
                                                          get_description_similarity(
                                                              x['search_term_beforethekey_thekey_stemmed'],
                                                              x['product_description_stemmed'], \
                                                              similarity_dict[
                                                                  x['search_term_beforethekey_thekey_stemmed']]),
                                                      axis=1)

df_all['description_similarity_10'] = df_all['description_similarity_tuple'].map(lambda x: x[0] / 10.0)
df_all['description_similarity_20'] = df_all['description_similarity_tuple'].map(lambda x: x[1] / 20.0)
df_all['description_similarity_11-20'] = df_all['description_similarity_tuple'].map(lambda x: (x[1] - x[0]) / 10.0)
df_all['description_similarity_30'] = df_all['description_similarity_tuple'].map(lambda x: x[2] / 30.0)
df_all['description_similarity_21-30'] = df_all['description_similarity_tuple'].map(lambda x: (x[2] - x[1]) / 10.0)

df_all['description_similarity_10rel'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * x[0] / x[3] if x[3] > 0.00001 else 0.0)
df_all['description_similarity_20rel'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * x[1] / x[4] if x[4] > 0.00001 else 0.0)
df_all['description_similarity_11-20rel'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * (x[1] - x[0]) / (x[4] - x[3]) if (x[4] - x[3]) > 0.00001 else 0.0)
df_all['description_similarity_30rel'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * x[2] / x[5] if x[5] > 0.00001 else 0.0)
df_all['description_similarity_21-30rel'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * (x[2] - x[1]) / (x[5] - x[4]) if (x[5] - x[4]) > 0.00001 else 0.0)

df_all['description_similarity_11-20to10'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * x[4] / x[3] if x[3] > 0.00001 else 0.0)
df_all['description_similarity_21-30to10'] = df_all['description_similarity_tuple'].map(
    lambda x: 1.0 * x[5] / x[3] if x[3] > 0.00001 else 0.0)

# this was added later, so this var is saved separately
df_all['above15_dummy_frequency_of_beforethekey_thekey'] = df_all['search_term_beforethekey_thekey_stemmed'].map(
    lambda x: int(similarity_dict[x]['cnt'] >= 15 and x != "_"))

t0 = time()
aa = list(set(list(df_all['search_term_beforethekey_thekey_stemmed'])))
my_dict = {}
for i in range(0, len(aa)):
    my_dict[aa[i]] = df_all['description_similarity_20'][df_all['search_term_beforethekey_thekey_stemmed'] == aa[i]]
    if i % 200 == 0:
        print i, "out of", len(aa), ";", round((time() - t0) / 60, 1), 'minutes'


def get_percentile_similarity(similarity20, str_thekey_pair, dict_item):
    output = 0.5
    N = len(dict_item)
    if str_thekey_pair != "_" and N >= 8:
        n = sum(dict_item >= similarity20)
        assert n > 0
        output = 1.0 * (n - 1) / (N - 1)
    return output


df_all['description20_percentile'] = df_all.apply(lambda x: \
                                                      get_percentile_similarity(x['description_similarity_20'], x[
                                                          'search_term_beforethekey_thekey_stemmed'], \
                                                                                my_dict[x[
                                                                                    'search_term_beforethekey_thekey_stemmed']]),
                                                  axis=1)

# these vars were added later, so they are saved separately
df_all[['id', 'above15_dummy_frequency_of_beforethekey_thekey', 'description20_percentile']].to_csv(
    FEATURES_DIR + "/df_feature_above15_ext.csv", index=False)
df_all = df_all.drop(['above15_dummy_frequency_of_beforethekey_thekey', 'description20_percentile'], axis=1)

print 'create description similarity variables time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()

df_all = df_all.drop(['description_similarity_tuple'], axis=1)

df_all = df_all.drop(['search_term_beforethekey_thekey_stemmed'], axis=1)

df_all.drop(string_variables_list, axis=1).to_csv(FEATURES_DIR + "/df_basic_features.csv", index=False)
print 'save file time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()
'''

#################################################################
### STEP 10: Brand and material dummies
#################################################################

df_all = df_all[['id'] + string_variables_list]

brand_dict = {}
material_dict = {}
import csv

with open(PROCESSINGTEXT_DIR + '/brand_statistics.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        brand_dict[row['name']] = {'cnt_attribute': int(row['cnt_attribute']), 'cnt_query': int(row['cnt_query'])}

with open(PROCESSINGTEXT_DIR + '/material_statistics.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        material_dict[row['name']] = {'cnt_attribute': int(row['cnt_attribute']), 'cnt_query': int(row['cnt_query'])}


# def create_dummy_value(lst,item):
#    return int(item in lst) * int(len(item)>1)


def col_create_brandmaterial_dummy(col_brandmaterial_txt, bm):
    bm_in_text = col_brandmaterial_txt.map(lambda x: x.split(';'))
    return bm_in_text.map(lambda x: int(bm in x))


BM_THRESHOLD = 500
cnt = 0
cnt1 = 0
for brand in brand_dict.keys():
    cnt += 1
    if brand_dict[brand]['cnt_attribute'] > BM_THRESHOLD:
        cnt1 += 1
        df_all["BRANDdummyintitle_" + brand.replace(" ", "")] = \
            col_create_brandmaterial_dummy(df_all['brands_in_product_title'], brand)
    if brand_dict[brand]['cnt_query'] > BM_THRESHOLD:
        cnt1 += 1
        df_all["BRANDdummyinquery_" + brand.replace(" ", "")] = \
            col_create_brandmaterial_dummy(df_all['brands_in_search_term'], brand)
    if cnt % 200 == 0:
        print cnt, "brands processed;", cnt1, "dummies created"

cnt = 0
cnt1 = 0
for material in material_dict.keys():
    cnt += 1
    if material_dict[material]['cnt_attribute'] > BM_THRESHOLD:
        cnt1 += 1
        df_all["MATERIALdummyintitle_" + material.replace(" ", "")] = \
            col_create_brandmaterial_dummy(df_all['materials_in_product_title'], material)
    if material_dict[material]['cnt_query'] > BM_THRESHOLD:
        cnt1 += 1
        df_all["MATERIALdummyinquery_" + material.replace(" ", "")] = \
            col_create_brandmaterial_dummy(df_all['materials_in_search_term'], material)
    if cnt % 200 == 0:
        print cnt, "materials processed;", cnt1, "dummies created"

print 'create brand and material dummies time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()

df_all.drop(string_variables_list, axis=1).to_csv(FEATURES_DIR + "/df_brand_material_dummies.csv", index=False)
print 'save file time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()

#################################################################
### STEP 11: Thekey dummies
#################################################################

df_all = df_all[['id'] + string_variables_list]

THEKEY_THRESHOLD = 500
lst = list(df_all['product_title_thekey_stemmed'])
freqs = [(w, lst.count(w)) for w in set(lst)]
cnt = 0
title_keys = []
for tpl in sorted(freqs, key=lambda x: x[1], reverse=True):
    if tpl[1] > THEKEY_THRESHOLD:
        cnt += 1
        title_keys.append(tpl[0])
        df_all["THEKEYdummyintitle_" + tpl[0]] = df_all['product_title_thekey_stemmed'].map(lambda x: int(x == tpl[0]))
print cnt, "product_title thekey dummies created"

lst = list(df_all['search_term_thekey_stemmed'])
freqs = [(w, lst.count(w)) for w in set(lst)]
cnt = 0
query_keys = []
for tpl in sorted(freqs, key=lambda x: x[1], reverse=True):
    if tpl[1] > THEKEY_THRESHOLD:
        cnt += 1
        query_keys.append(tpl[0])
        df_all["THEKEYdummyinquery_" + tpl[0]] = df_all['search_term_thekey_stemmed'].map(lambda x: int(x == tpl[0]))
print cnt, "search_term thekey dummies created"

BEFORETHEKEYTHEKEY_THRESHOLD = 300
lst = list(df_all['product_title_beforethekey_stemmed'] + "_" + df_all['product_title_thekey_stemmed'])
freqs = [(w, lst.count(w)) for w in set(lst)]
cnt = 0
for tpl in sorted(freqs, key=lambda x: x[1], reverse=True):
    if tpl[1] > BEFORETHEKEYTHEKEY_THRESHOLD:
        cnt += 1
        df_all["BTK_TKdummyintitle_" + tpl[0]] = (
        df_all['product_title_beforethekey_stemmed'] + "_" + df_all['product_title_thekey_stemmed']).map(
            lambda x: int(x == tpl[0]))
print cnt, "product_title beforethekey_thekey dummies created"

lst = list(df_all['search_term_beforethekey_stemmed'] + "_" + df_all['search_term_thekey_stemmed'])
freqs = [(w, lst.count(w)) for w in set(lst)]
cnt = 0
for tpl in sorted(freqs, key=lambda x: x[1], reverse=True):
    if tpl[1] > BEFORETHEKEYTHEKEY_THRESHOLD:
        cnt += 1
        df_all["BTK_TKdummyinquery_" + tpl[0]] = (
        df_all['search_term_beforethekey_stemmed'] + "_" + df_all['search_term_thekey_stemmed']).map(
            lambda x: int(x == tpl[0]))
print cnt, "search_term beforethekey_thekey dummies created"

print 'create thekeys dummies time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()

df_all.drop(string_variables_list, axis=1).to_csv(FEATURES_DIR + "/df_thekey_dummies.csv", index=False)
print 'save file time:', round((time() - t0) / 60, 1), 'minutes\n'
t0 = time()

print 'TOTAL FEATURE EXTRACTION TIME:', round((time() - t1) / 60, 1), 'minutes\n'



################################

