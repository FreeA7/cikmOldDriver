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

# 9. 将计量单位的表示统一
# ./Code/Chenglong/data_processor, line 181 - 209
class UnitConverter(BaseReplacer):
    """
    shadeMature height: 36 in. - 48 in.Mature width
    PUT one UnitConverter before LowerUpperCaseSplitter
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. "),
            (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. "),
            (r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\1 sq.in. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. "),
            (r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. "),
            (r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. "),
            (r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. "),
            (r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. "),
            (r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. "),
            (r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. "),
            (r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. "),
            (r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. "),
            (r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. "),
            (r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. "),
            (r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr "),
            (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. "),
            (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr "),
        ]

# 10. 特殊字符处理，用于完善test.py 中的 line 73 - 78
# ./Code/Chenglong/data_processor, line 226 - 246
class QuartetCleaner(BaseReplacer):

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"<.+?>", r""),
            # html codes
            (r"&nbsp;", r" "),
            (r"&amp;", r"&"),
            (r"&#39;", r"'"),
            (r"/>/Agt/>", r""),
            (r"</a<gt/", r""),
            (r"gt/>", r""),
            (r"/>", r""),
            (r"<br", r""),
            # do not remove [".", "/", "-", "%"] as they are useful in numbers,
            # e.g., 1.97, 1-1/2, 10%, etc.
            (r"[ &<>)(_,;:!?\+^~@#\$]+", r" "),
            ("'s\\b", r""),
            (r"[']+", r""),
            (r"[\"]+", r""),
        ]

# 11. Stemming以及Lemmatizing
# ./Code/Chenglong/data_processor, line 251 - 275
# lemmatizing for using pretrained word2vec model
# 2nd solution in CrowdFlower
class Lemmatizer:

    def __init__(self):
        self.Tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.Lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def transform(self, text):
        tokens = [self.Lemmatizer.lemmatize(
            token) for token in self.Tokenizer.tokenize(text)]
        return " ".join(tokens)


# stemming
class Stemmer:

    def __init__(self, stemmer_type="snowball"):
        self.stemmer_type = stemmer_type
        if self.stemmer_type == "porter":
            self.stemmer = nltk.stem.PorterStemmer()
        elif self.stemmer_type == "snowball":
            self.stemmer = nltk.stem.SnowballStemmer("english")

    def transform(self, text):
        tokens = [self.stemmer.stem(token) for token in text.split(" ")]
        return " ".join(tokens)

# 12. Igor的单词替换，和5对应
# ./Code/Igor&Kostia/homedepot_functions, line 241 - 332
def replace_in_parser(s):
    #the first three shared on forum
    s=s.replace("acccessories","accessories")
    s = re.sub(r'\bscott\b', 'scotts', s) #brand
    s = re.sub(r'\borgainzer\b', 'organizer', s)
    
    # the others are not shared
    s = re.sub(r'\aluminuum\b', 'aluminum', s)    
    s = re.sub(r'\bgeneral electric','ge', s)
    s = s.replace("adaptor","adapter")
    s = re.sub(r'\bfibre', 'fiber', s)
    s = re.sub(r'\bbuilt in\b', 'builtin',s)
    s = re.sub(r'\bshark bite\b', 'sharkbite',s)
    s = re.sub('barbeque', 'barbecue',s)
    s = re.sub(r'\bbbq\b', 'barbecue', s)
    s = re.sub(r'\bbathroom[s]*\b', 'bath', s)
    s = re.sub(r'\bberkeley\b', 'berkley', s)
    s = re.sub(r'\bbookshelves\b', 'book shelf', s)
    s = re.sub(r'\bbookshelf\b', 'book shelf', s)
    s = re.sub(r'\bin line ', ' inline ', s)
    s = re.sub(r'round up\b', ' roundup', s)
    s = re.sub(r'\blg electronics\b', 'lg', s)
    s = re.sub(r'\bhdtv\b', 'hd tv', s)
    s = re.sub(r'black [and ]*decker', 'black and decker', s)
    s = re.sub(r'backer board[s]*', 'backerboard', s)
    s = re.sub(r'\bphillips\b', 'philips', s)
    s = re.sub(r'\bshower head[s]*\b', 'showerhead', s)
    s = re.sub(r'\bbull nose\b', 'bullnose', s)
    s = re.sub(r'\bflood light\b', 'floodlight', s)
    s = re.sub(r'\barrester\b', 'arrestor', s)
    s = re.sub(r'\bbi fold\b', 'bifold', s)
    s = re.sub(r'\bfirepit[s]*\b', 'fire pit', s)
    s = re.sub(r'\bbed bug[s]*\b', 'bedbug', s)
    s = re.sub(r'\bhook up[s]*\b', 'hookup', s)
    s = re.sub(r'\bjig saw[s]*\b', 'jigsaw', s)
    s = re.sub(r'\bspacesav(?=er[s]*|ing)', 'space sav', s)
    s = re.sub(r'\bwall paper', 'wallpaper', s)
    s = re.sub(r'\bphotocell', 'photo cells', s)
    s = re.sub(r'\bplasti dip\b', 'plastidip', s)
    s = re.sub(r'\bflexi dip\b', 'flexidip', s)  
    s = re.sub(r'\bback splash','backsplash', s)
    s = re.sub(r'\bbarstool(?=\b|s)','bar stool', s)
    s = re.sub(r'\blampholder(?=\b|s)','lamp holder', s)
    s = re.sub(r'\brainsuit(?=\b|s)','rain suit', s)
    s = re.sub(r'\bback up\b','backup', s)
    s = re.sub(r'\bwheel barrow', 'wheelbarrow', s)
    s=re.sub(r'\bsaw horse', 'sawhorse',s)
    s=re.sub(r'\bscrew driver', 'screwdriver',s)
    s=re.sub(r'\bnut driver', 'nutdriver',s)
    s=re.sub(r'\bflushmount', 'flush mount',s)
    s=re.sub(r'\bcooktop(?=\b|s\b)', 'cook top',s)
    s=re.sub(r'\bcounter top(?=s|\b)','countertop', s)    
    s=re.sub(r'\bbacksplash', 'back splash',s)
    s=re.sub(r'\bhandleset', 'handle set',s)
    s=re.sub(r'\bplayset', 'play set',s)
    s=re.sub(r'\bsidesplash', 'side splash',s)
    s=re.sub(r'\bdownlight', 'down light',s)
    s=re.sub(r'\bbackerboard', 'backer board',s)
    s=re.sub(r'\bshoplight', 'shop light',s)
    s=re.sub(r'\bdownspout', 'down spout',s)
    s=re.sub(r'\bpowerhead', 'power head',s)
    s=re.sub(r'\bnightstand', 'night stand',s)
    s=re.sub(r'\bmicro fiber[s]*\b', 'microfiber', s)
    s=re.sub(r'\bworklight', 'work light',s)
    s=re.sub(r'\blockset', 'lock set',s)
    s=re.sub(r'\bslatwall', 'slat wall',s)
    s=re.sub(r'\btileboard', 'tile board',s)
    s=re.sub(r'\bmoulding', 'molding',s)
    s=re.sub(r'\bdoorstop', 'door stop',s)
    s=re.sub(r'\bwork bench\b','workbench', s)
    s=re.sub(r'\bweed[\ ]*eater','weed trimmer', s)
    s=re.sub(r'\bweed[\ ]*w[h]*acker','weed trimmer', s)
    s=re.sub(r'\bnightlight(?=\b|s)','night light', s)
    s=re.sub(r'\bheadlamp(?=\b|s)','head lamp', s)
    s=re.sub(r'\bfiber board','fiberboard', s)
    s=re.sub(r'\bmail box','mailbox', s)
    
    replace_material_dict={'aluminium': 'aluminum', 
    'medium density fiberboard': 'mdf',
    'high density fiberboard': 'hdf',
    'fiber reinforced polymer': 'frp',
    'cross linked polyethylene': 'pex',
    'poly vinyl chloride': 'pvc', 
    'thermoplastic rubber': 'tpr', 
    'poly lactic acid': 'pla', 
    'acrylonitrile butadiene styrene': 'abs',
    'chlorinated poly vinyl chloride': 'cpvc'}
    for word in replace_material_dict.keys():
        if word in s:
            s = s.replace(word, replace_material_dict[word])
    
    return s

# 13. Igor的计量单位替换，和9对应，以及很多其他的替换
# ./Code/Igor&Kostia/homedepot_functions, line 338 - 523
def str_parser(s, automatic_spell_check_dict={}, remove_from_brackets=False,parse_material=False,add_space_stop_list=[]):
    #the following three replacements are shared on the forum    
    s = s.replace("craftsm,an","craftsman")        
    s = re.sub(r'depot.com/search=', '', s)
    s = re.sub(r'pilers,needlenose', 'pliers, needle nose', s)
    
    s = re.sub(r'\bmr.', 'mr ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = re.sub(r'(?<=[0-9]),[\ ]*(?=[0-9])', '', s)
    s = s.replace(";",".")
    s = s.replace(",",".")
    s = s.replace(":",". ")
    s = s.replace("+"," ")
    s = re.sub(r'\bU.S.', 'US ', s)
    s = s.replace(" W x "," ")
    s = s.replace(" H x "," ")
    s = re.sub(' [\#]\d+[\-\d]*[\,]*', '', s)    
    s = re.sub('(?<=[0-9\%])(?=[A-Z][a-z])', '. ', s) # add dot between number and cap letter
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s) # add space between parentheses and letters

    if parse_material:
        replace_dict={'Medium Density Fiberboard (MDF)':'mdf', 'High Density Fiberboard (HDF)':'hdf',\
        'Fibre Reinforced Polymer (FRP)': 'frp', 'Acrylonitrile Butadiene Styrene (ABS)': 'abs',\
        'Cross-Linked Polyethylene (PEX)':'pex', 'Chlorinated Poly Vinyl Chloride (CPVC)': 'cpvc',\
        'PVC (vinyl)': 'pvc','Thermoplastic rubber (TPR)':'tpr','Poly Lactic Acid (PLA)': 'pla',\
        '100% Polyester':'polyester','100% UV Olefin':'olefin', '100% BCF Polypropylene': 'polypropylene',\
        '100% PVC':'pvc'}
        
        if s in replace_dict.keys():
            s=replace_dict[s]


    s = re.sub('[^a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]', ' ', s)
    s= " ".join(s.split())

    s=s.replace("-"," ")
    
    if len(add_space_stop_list)>0:
        s = " ".join([re.sub('(?<=[a-z])(?=[A-Z][a-z\ ])', '. ', word)  if word.lower() not in add_space_stop_list else word for word in s.split()])

    s=s.lower() 
    s = re.sub('\.(?=[a-z])', '. ', s) #dots before words -> replace with spaces
   # s = re.sub('(?<=[a-z])(?=[A-Z][a-z\ ])', ' ', s) # add space if uppercase after lowercase
    s = re.sub('(?<=[a-z][a-z][a-z])(?=[0-9])', ' ', s) # add cpase if number after at least three letters
    ##s = re.sub('(?<=[a-zA-Z])\.(?=\ |$)', '', s) #remove dots at the end of string
    #s = re.sub('(?<=[0-9])\.(?=\ |$)', '', s) # dot after digit before space
    s = re.sub('^\.\ ', '', s) #dot at the beginning before space
    

    if len(automatic_spell_check_dict.keys())>0:
        s=spell_correction(s,automatic_spell_check_dict=automatic_spell_check_dict)
    
    if remove_from_brackets==True:
        s = re.sub('(?<=\()[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*(?=\))', '', s)
    else:
        s=s.replace(" (",". ")
        s=re.sub('(?<=[a-zA-Z0-9\%\$])\(', '. ', s)
        s=s.replace(" )",". ")
        s=s.replace(")",". ")
        s=s.replace("  "," ")
        s = re.sub('\ \.', '\.', s)
        

    #######s = re.sub('(?<=[0-9\%])(?=[a-wyz])', ' ', s) # add space between number and text (except letter x) 
    #s = re.sub('(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', s) # replace '-' in words with space
    s=s.replace("at&t","att")
    s=s.replace("&"," and ")    
    s=s.replace("*"," x ")
    s = re.sub('(?<=[a-z\ ])\/(?=[a-z\ ])', ' ', s) # replace "/" between words with space
    s = re.sub('(?<=[a-z])\\\\(?=[a-z])', ' ', s) # replace "/" between words with space
    s=s.replace("  "," ")
    s=s.replace("  "," ")
    
    #s=re.sub('(?<=\ [a-ux-z])\ (?=[0-9])', '', s)   #remove spaces
    #s=re.sub('(?<=^[a-z])\ (?=[0-9])', '', s)   #remove spaces




    #####################################
    ### thesaurus replacement in all vars
    s=replace_in_parser(s)
    
    s = re.sub('half(?=\ inch)', '1/2', s)
    s = re.sub('\ba half\b', '1/2', s)
    #s = re.sub('half\ ', 'half-', s)

    s = re.sub(r'(?<=\')s\b', '', s)
    s = re.sub('(?<=[0-9])\'\'', ' in ', s)
    s = re.sub('(?<=[0-9])\'', ' in ', s)

    s = re.sub(r'(?<=[0-9])[\ ]*inch[es]*\b', '-in ', s)
    s = re.sub(r'(?<=[0-9])[\ ]*in\b', '-in ', s)
    
    s = re.sub(r'(?<=[0-9])[\-|\ ]*feet[s]*\b', '-ft ', s)
    s = re.sub(r'(?<=[0-9])[\ ]*foot[s]*\b', '-ft ', s)
    s = re.sub(r'(?<=[0-9])[\ ]*ft[x]*\b', '-ft ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*volt[s]*(?=\ |$|\.)', '-V ', s)
    s = re.sub('(?<=[0-9])[\ ]*v(?=\ |$|\.)', '-V ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*wat[t]*[s]*(?=\ |$|\.)', '-W ', s)
    s = re.sub('(?<=[0-9])[\ ]*w(?=\ |$|\.)', '-W ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*kilo[\ ]*watt[s]*(?=\ |$|\.)', '-KW ', s)
    s = re.sub('(?<=[0-9])[\ ]*kw(?=\ |$|\.)', '-KW ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*amp[s]*(?=\ |$|\.)', '-A ', s)
    #s = re.sub('(?<=[0-9]) a(?=\ |$|\.)', '-A. ', s)
    s = re.sub('(?<=[0-9])a(?=\ |$|\.)', '-A ', s)

    s = re.sub('(?<=[0-9])[\ ]*gallon[s]*(?=\ |$|\.)', '-gal ', s)
    s = re.sub('(?<=[0-9])[\ ]*gal(?=\ |$|\.)', '-gal ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*pound[s]*(?=\ |$|\.)', '-lb ', s)
    s = re.sub('(?<=[0-9])[\ ]*lb[s]*(?=\ |$|\.)', '-lb ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*mi[l]+imet[er]*[s]*(?=\ |$|\.)', '-mm ', s)
    s = re.sub('(?<=[0-9])[\ ]*mm(?=\ |$|\.)', '-mm ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*centimeter[s]*(?=\ |$|\.)', '-cm ', s)
    s = re.sub('(?<=[0-9])[\ ]*cm(?=\ |$|\.)', '-cm ', s)
        
    s = re.sub('(?<=[0-9])[\ ]*ounce[s]*(?=\ |$|\.)', '-oz ', s)
    s = re.sub('(?<=[0-9])[\ ]*oz(?=\ |$|\.)', '-oz ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*liter[s]*(?=\ |$|\.)', '-L ', s)
    s = re.sub('(?<=[0-9])[\ ]*litre[s]*(?=\ |$|\.)', '-L ', s)
    s = re.sub('(?<=[0-9])[\ ]*l(?=\ |$|\.)', '-L. ', s)
    
    s = re.sub('(?<=[0-9])[\ ]*square feet[s]*(?=\ |$|\.)', '-sqft ', s)
    s = re.sub('(?<=[0-9])square feet[s]*(?=\ |$|\.)', '-sqft ', s)
    s = re.sub('(?<=[0-9])[\ ]*sq[\ |\.|\.\ ]*ft(?=\ |$|\.)', '-sqft ', s)
    s = re.sub('(?<=[0-9])[\ ]*sq. ft(?=\ |$|\.)', '-sqft', s)
    s = re.sub('(?<=[0-9])[\ ]*sq.ft(?=\ |$|\.)', '-sqft', s)
    
    s = re.sub('(?<=[0-9])[\ ]*cubic f[e]*t[s]*(?=\ |$|\.)', '-cuft ', s)
    s = re.sub('(?<=[0-9])[\ ]*cu[\ |\.|\.\ ]*ft(?=\ |$|\.)', '-cuft ', s)
    s = re.sub('(?<=[0-9])[\ ]*cu[\.]*[\ ]*ft(?=\ |$|\.)', '-cuft', s)
    
     
    #remove 'x'
    s = re.sub('(?<=[0-9]) x (?=[0-9])', '-X ', s)
    s = re.sub('(?<=[0-9])x (?=[0-9])', '-X ', s)
    s = re.sub('(?<=[0-9]) x(?=[0-9])', '-X ', s)
    s = re.sub('(?<=[0-9])x(?=[0-9])', '-X ', s)
    
    #s=s.replace("..",".")
    s=s.replace("\n"," ")
    s=s.replace("  "," ")

    words=s.split()

    if s.find("-X")>=0:
        for cnt in range(0,len(words)-1):
            if words[cnt].find("-X")>=0:
                if words[cnt+1].find("-X") and cnt<len(words)-2:
                    cntAdd=2
                else:
                    cntAdd=1
                to_replace=re.search(r'(?<=[0-9]\-)\w+\b',words[cnt+cntAdd])
                if not (to_replace==None):
                    words[cnt]=words[cnt].replace("-X","-"+to_replace.group(0)+"")
                else:
                    words[cnt]=words[cnt].replace("-X","x")
    s = " ".join([word for word in words])
    
    s = re.sub('[^a-zA-Z0-9\ \%\$\-\@\&\/\.]', '', s) #remove "'" and "\n" and "#" and characters
    ##s = re.sub('(?<=[a-zA-Z])[\.|\/](?=\ |$)', '', s) #remove dots at the end of string
    s = re.sub('(?<=[0-9])x(?=\ |$)', '', s) #remove 
    s = re.sub('(?<=[\ ])x(?=[0-9])', '', s) #remove
    s = re.sub('(?<=^)x(?=[0-9])', '', s)
    #s = re.sub('[\ ]\.(?=\ |$)', '', s) #remove dots 
    s=s.replace("  "," ")
    s=s.replace("..",".")
    s = re.sub('\ \.', '', s)
    
    s=re.sub('(?<=\ [ch-hj-np-su-z][a-z])\ (?=[0-9])', '', s) #remove spaces
    s=re.sub('(?<=^[ch-hj-np-su-z][a-z])\ (?=[0-9])', '', s) #remove spaces
    
    s = re.sub('(?<=\ )\.(?=[0-9])', '0.', s)
    s = re.sub('(?<=^)\.(?=[0-9])', '0.', s)
    return " ".join([word for word in s.split()])

# 14. Igor的Stemming，用的是nltk下的SnowballStemmer，和11对应
# ./Code/Igor&Kostia/homedepot_functions, line 531 - 539
def str_stemmer(s, automatic_spell_check_dict={},remove_from_brackets=False,parse_material=False,add_space_stop_list=[], stoplist=stoplist):
    s=str_parser(s,automatic_spell_check_dict=automatic_spell_check_dict, remove_from_brackets=remove_from_brackets,\
            parse_material=parse_material, add_space_stop_list=add_space_stop_list)
    s=" ".join([word for word in s.split() if word not in stoplist])
    return " ".join([stemmer.stem(re.sub('\.(?=$)', '', word)) for word in s.split()])

def str_stemmer_wo_parser(s, stoplist=stoplist):
    s=" ".join([word for word in s.split() if word not in stoplist])
    return " ".join([stemmer.stem(re.sub('\.(?=$)', '', word)) for word in s.split()])

# 15. 用 Google dictionary修正拼写，代码见 https://www.kaggle.com/steubk/fixing-typos 以及：
# ./Code/Chenglong/google_spelling_checker_dict.py
# ./Code/Igor&Kostia/google_dict.py
#
# 16. 人工替换部分词语，单词表在：
# ./Data/dict/word_replacer.csv
#
# 17. 加入外部数据？ 用处尚不清楚，见：
# ./Data/dict/color_data.py
#
# 18. 同义词替换，见：
# ./Data/dict/word_replacer.csv, line 197
#
# 19. 缩写替换，见：
# ./Data/dict/word_replacer.csv, line 211
# ./Code/Igor&Kostia/homedepot_functions.py, line 318 - 327, 361 - 367
#
# 20. 拼写统一，比如 mailbox 和 mail box, fibre 和 fiber 等, 见：
# ./Code/Igor&Kostia/homedepot_functions.py, line 83 - 316
#
# 21. Part-of-Speech Tagging
# 用 NLTK.pos_tagger() 函数
#
# 22. 材质和品牌名称的替换，19中也有涉及，将太长的品牌名称缩短，减少无效信息
# ./Code/Igor&Kostia/homedepot_functions.py, line 318 - 327, 361 - 367
# ./Code/Igor&Kostia/text_processing.py, line 331 - 471
#
# 23. 对产品名称的不同部分分别作处理，找出产品名中最重要的部分，比如应该区别对待产品名中 with, for, in, without 之后的词
# 见：http://blog.kaggle.com/2015/07/22/crowdflower-winners-interview-3rd-place-team-quartet/
# 论文对应原文：
#   We noticed a structure in product title which helped us to find the most important
#   part of the document. For example, in product title 'Husky 52 in. 10-Drawer Mobile
#   Workbench with Solid Wood Top, Black' the product is workbench, not wood top. To
#   deal with multiple patterns present in product title, we had to elaborate a complex
#   algorithm, which extracted important words.
#   We also extracted the top trigram. Igor and Kostia started this work indepen-
#   dently, so in their notation the trigrams words are denoted as (before2thekey, before-
#   thekey, thekey), where thekey is the last word in the trigram.
#   We separately dealt with text after words with, for, in, without.
#
# 24. 将小写单词与大写单词分隔，把商品描述中遗漏的空格补上，类似于 lower[.?!]UPPER 这种情况
# ./Code/Chenglong/data_processor.py, line 52 - 74
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

# 25. 英文数字变成阿拉伯数字
# ./Code/Chenglong/data_processor.py, line 151 - 168
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
            (r"(?<=\W|^)%s(?=\W|$)"%n, str(d)) for n,d in zip(numbers, digits)
        ]

# 26. 作者自己编写的拼写检查
# ./Code/Chenglong/spelling_checker.py, line 34 - 315
# class PeterNovigSpellingChecker 和 class AutoSpellingChecker:
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
