# -*- coding: utf-8 -*-
"""
@author: Yifan Tang
@time: 2017/7/4 11:20 AM
"""

# 1. 去掉括号以及括号中的内容（已经写了）

# 2. 移除数字中的逗号，如：1,000 -> 1000
# ./Code/Chenglong/data_processor, line 148 - 156
class DigitCommaDigitMerger(BaseReplacer):
    """
    1,000,000 -> 1000000
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(?<=\d+),(?=000)", r""),
        ]

# 3. 大写字母统一变小写
# ./Code/Chenglong/data_processor, line 46 - 52
class LowerCaseConverter(BaseReplacer):
    """
    Traditional -> traditional
    """

    def transform(self, text):
        return text.lower()

# 4. 句内出现“单词小写字母结尾+下一单词大写字母开头”形式，则在大小写间加空格
# ./Code/Chenglong/data_processor, line 55 - 78
class LowerUpperCaseSplitter(BaseReplacer):
    """
    homeBASICS Traditional Real Wood -> homeBASICS Traditional Real Wood

    hidden from viewDurable rich finishLimited lifetime warrantyEncapsulated panels ->
    hidden from view Durable rich finish limited lifetime warranty Encapsulated panels

    Dickies quality has been built into every product.Excellent visibilityDurable ->
    Dickies quality has been built into every product Excellent visibility Durable

    BAD CASE:
    shadeMature height: 36 in. - 48 in.Mature width
    minutesCovers up to 120 sq. ft.Cleans up
    PUT one UnitConverter before LowerUpperCaseSplitter

    Reference:
    https://www.kaggle.com/c/home-depot-product-search-relevance/forums/t/18472/typos-in-the-product-descriptions
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(\w)[\.?!]([A-Z])", r"\1 \2"),
            (r"(?<=( ))([a-z]+)([A-Z]+)", r"\2 \3"),
        ]

# 5. 单词替换，replace_fname = word_replacer.csv
# word_replacer.csv文件就在和本文同一目录之下
# 文件中网页上的内容与文件之间的关系有待核实
# ./Code/Chenglong/data_processor, line 83 - 97
class WordReplacer(BaseReplacer):

    def __init__(self, replace_fname):
        self.replace_fname = replace_fname
        self.pattern_replace_pair_list = []
        for line in csv.reader(open(self.replace_fname)):
            if len(line) == 1 and line[0].startswith("#"):
                continue
            try:
                pattern = r"(?<=\W|^)%s(?=\W|$)" % line[0]
                replace = line[1]
                self.pattern_replace_pair_list.append((pattern, replace))
            except:
                print(line)
                pass

# 6. 将“单词/单词”以及“单词-单词”分割，但是数字的这两种形式保留
# ./Code/Chenglong/data_processor, line 101 - 122
class LetterLetterSplitter(BaseReplacer):
    """
    For letter and letter
    /:
    Cleaner/Conditioner -> Cleaner Conditioner

    -:
    Vinyl-Leather-Rubber -> Vinyl Leather Rubber

    For digit and digit, we keep it as we will generate some features via math operations,
    such as approximate height/width/area etc.
    /:
    3/4 -> 3/4

    -:
    1-1/4 -> 1-1/4
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2"),
        ]

# 7. 处理“字符+标点”形式
# ./Code/Chenglong/data_processor, line 126 - 145
class DigitLetterSplitter(BaseReplacer):
    """
    x:
    1x1x1x1x1 -> 1 x 1 x 1 x 1 x 1
    19.875x31.5x1 -> 19.875 x 31.5 x 1

    -:
    1-Gang -> 1 Gang
    48-Light -> 48 Light

    .:
    includes a tile flange to further simplify installation.60 in. L x 36 in. W x 20 in. ->
    includes a tile flange to further simplify installation. 60 in. L x 36 in. W x 20 in.
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2"),
            (r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2"),
        ]


# 8. 将英文数字替换成阿拉伯数字
# ./Code/Chenglong/data_processor, line 159 - 177
class NumberDigitMapper(BaseReplacer):
    """
    one -> 1
    two -> 2
    """

    def __init__(self):
        numbers = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90
        ]
        self.pattern_replace_pair_list = [
            (r"(?<=\W|^)%s(?=\W|$)" % n, str(d)) for n, d in zip(numbers, digits)
        ]

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# 3. 在数字（左边）和字母（右边）之间加空格，如：5inch -> 5 inch
# ./Code/Igor&Kostia/homedepot_functions, line 404
s = re.sub('(?<=[0-9\%])(?=[a-wyz])', ' ', s)

# 4. 在至少三个字母（左边）和数字（右边）之间加空格，如color5 -> color 5
# 
