import re
import nltk
import requests

'''
15 google太慢，baidu好多检查不出来，所以用bing
17 不知道干什么所以还没加
21 chenglong的代码里说明了可以用但是他们并没有用，我先没有加
23 不懂，我们需要讨论一下
'''

# 移除数字中的逗号 - 2
def digitcommadigitmerger(text):
    """
    1,000,000 -> 1000000
    """
    # 设定数字匹配模式
    digitpattern = re.compile(r'[0-9]+,[0-9]{3}')

    # 需要进行两次
    # 1,000,000,000 --> 1000,000000
    # 1000,000000 --> 1000000000
    for i in range(2):
        digittext = digitpattern.findall(text)
        if digittext:
            for texts in digittext:
                text = text.replace(texts, texts.replace(',',''))
    return text

# 大写字母统一变小写 - 3
def lowercaseconverter(text):
    """
    Traditional -> traditional
    """

    return text.lower()

# “单词小写字母结尾+下一单词大写字母开头”形式，在大小写间加空格 - 4 - 24
def loweruppercasesplitter(text):
    text = re.sub(r"(\w)[\.?!]([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z]+)([A-Z]+)", r"\1 \2", text)
    return text

# 单词替换 - 5 - 16 - 18 - 19
class WordReplacer():

    def __init__(self):
        self.replacedict = {}
        f = open('./data/setting/word_replacer.csv')
        while 1:
            line = f.readline()
            if line:
                if line[0] == '#':
                    continue
                else:
                    self.replacedict[line[:line.find(',')]] = line[line.find(',')+1:-1]
            else:
                break

    def replace(self, text):
        for word in self.replacedict.keys():
            while word in text:
                text = text.replace(word, self.replacedict[word])
        return text

# 将“单词/单词”以及“单词-单词”分割，但是数字的这两种形式保留 - 6
def letterlettersplitter(text):
    text = re.sub(r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2", text)
    return text

# 处理“字符+标点”形式 - 7
def digitlettersplitter(text):
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

    text = re.sub(r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2", text)

# 将英文数字替换成阿拉伯数字 - 8 - 25
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

    def replace(self, text):
        for pattern in self.pattern_replace_pair_list:
            text = re.sub(pattern[0], pattern[1], text)
        return text

# 将计量单位的表示统一 - 9
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

    def replace(self, text):
        for pattern in self.pattern_replace_pair_list:
            text = re.sub(pattern[0], pattern[1], text)
        return text

# Stemming以及Lemmatizing - 11
class Lemmatizer:

    def __init__(self):
        self.Tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.Lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def transform(self, text):
        tokens = [self.Lemmatizer.lemmatize(
            token) for token in self.Tokenizer.tokenize(text)]
        return " ".join(tokens)

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

# Igor的单词替换 - 12 - 19 - 20 - 22
def igor_replace_in_parser(text):
    s = text

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
        while word in s:
            s = s.replace(word, replace_material_dict[word])
    text = s
    return text

# Igor的各种替换 - 13 - 19 - 20 - 22
def igor_str_parser(text, automatic_spell_check_dict={}, remove_from_brackets=False,parse_material=False,add_space_stop_list=[]):
    s = text

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

# Igor的Stemming - 14
def igor_str_stemmer(s, automatic_spell_check_dict={},remove_from_brackets=False,parse_material=False,add_space_stop_list=[], stoplist=stoplist):
    s=str_parser(s,automatic_spell_check_dict=automatic_spell_check_dict, remove_from_brackets=remove_from_brackets,\
            parse_material=parse_material, add_space_stop_list=add_space_stop_list)
    s=" ".join([word for word in s.split() if word not in stoplist])
    return " ".join([stemmer.stem(re.sub('\.(?=$)', '', word)) for word in s.split()])

def igor_str_stemmer_wo_parser(s, stoplist=stoplist):
    s=" ".join([word for word in s.split() if word not in stoplist])
    return " ".join([stemmer.stem(re.sub('\.(?=$)', '', word)) for word in s.split()])

# bing拼写检查 - 15 - 26
def spell_check(text):
    q = '+'.join(text.split())
    r = requests.get('http://cn.bing.com/search?q='+q)
    html = r.text
    html = html[html.find('<div id="sp_requery">'):]
    html = html[:html.find('</div>')]
    html = re.sub(r'[<][^>]+[>]', '', html)
    html = re.replace(' ','')
    html = re.replace('包含', '')
    html = re.replace('的结果。', '')
    return html

# 材质和品牌名称的替换 - 22
class BrandReplace():

### some brand names in "MFG Brand Name" of attributes.csv have a few words
### but it is much more likely for a person to search for brand 'BEHR' 
### than 'BEHR PREMIUM PLUS ULTRA'. That is why we replace long brand names 
### with a shorter alternatives
    def __init__(self):
        self.replace_brand_dict={
        'acurio latticeworks': 'acurio', 
        'american kennel club':'akc',
        'amerimax home products': 'amerimax',
        'barclay products':'barclay',
        'behr marquee': 'behr', 
        'behr premium': 'behr', 
        'behr premium deckover': 'behr', 
        'behr premium plus': 'behr', 
        'behr premium plus ultra': 'behr', 
        'behr premium textured deckover': 'behr', 
        'behr pro': 'behr', 
        'bel air lighting': 'bel air',
        'bootz industries':'bootz',
        'campbell hausfeld':'campbell',
        'columbia forest products': 'columbia',
        'essick air products':'essick air',
        'evergreen enterprises':'evergreen',
        'feather river doors': 'feather river', 
        'gardner bender':'gardner',
        'ge parts':'ge',
        'ge reveal':'ge',
        'gibraltar building products':'gibraltar',
        'gibraltar mailboxes':'gibraltar',
        'glacier bay':'glacier',
        'great outdoors by minka lavery': 'great outdoors', 
        'hamilton beach': 'hamilton',
        'hampton bay':'hampton',
        'hampton bay quickship':'hampton',
        'handy home products':'handy home',
        'hickory hardware': 'hickory', 
        'home accents holiday': 'home accents',
        'home decorators collection': 'home decorators',
        'homewerks worldwide':'homewerks',
        'klein tools': 'klein',
        'lakewood cabinets':'lakewood',
        'leatherman tool group':'leatherman',
        'legrand adorne':'legrand',
        'legrand wiremold':'legrand',
        'lg hausys hi macs':'lg',
        'lg hausys viatera':'lg',
        'liberty foundry':'liberty',
        'liberty garden':'liberty',
        'lithonia lighting':'lithonia',
        'loloi rugs':'loloi',
        'maasdam powr lift':'maasdam',
        'maasdam powr pull':'maasdam',
        'martha stewart living': 'martha stewart',
        'merola tile': 'merola',
        'miracle gro':'miracle',
        'miracle sealants':'miracle',
        'mohawk home': 'mohawk',
        'mtd genuine factory parts':'mtd',
        'mueller streamline': 'mueller',
        'newport coastal': 'newport',
        'nourison overstock':'nourison',
        'nourison rug boutique':'nourison',
        'owens corning': 'owens', 
        'premier copper products':'premier',
        'price pfister':'pfister',
        'pride garden products':'pride garden',
        'prime line products':'prime line',
        'redi base':'redi',
        'redi drain':'redi',
        'redi flash':'redi',
        'redi ledge':'redi',
        'redi neo':'redi',
        'redi niche':'redi',
        'redi shade':'redi',
        'redi trench':'redi',
        'reese towpower':'reese',
        'rheem performance': 'rheem',
        'rheem ecosense': 'rheem',
        'rheem performance plus': 'rheem',
        'rheem protech': 'rheem',
        'richelieu hardware':'richelieu',
        'rubbermaid commercial products': 'rubbermaid', 
        'rust oleum american accents': 'rust oleum', 
        'rust oleum automotive': 'rust oleum', 
        'rust oleum concrete stain': 'rust oleum', 
        'rust oleum epoxyshield': 'rust oleum', 
        'rust oleum flexidip': 'rust oleum', 
        'rust oleum marine': 'rust oleum', 
        'rust oleum neverwet': 'rust oleum', 
        'rust oleum parks': 'rust oleum', 
        'rust oleum professional': 'rust oleum', 
        'rust oleum restore': 'rust oleum', 
        'rust oleum rocksolid': 'rust oleum', 
        'rust oleum specialty': 'rust oleum', 
        'rust oleum stops rust': 'rust oleum', 
        'rust oleum transformations': 'rust oleum', 
        'rust oleum universal': 'rust oleum', 
        'rust oleum painter touch 2': 'rust oleum',
        'rust oleum industrial choice':'rust oleum',
        'rust oleum okon':'rust oleum',
        'rust oleum painter touch':'rust oleum',
        'rust oleum painter touch 2':'rust oleum',
        'rust oleum porch and floor':'rust oleum',
        'salsbury industries':'salsbury',
        'simpson strong tie': 'simpson', 
        'speedi boot': 'speedi', 
        'speedi collar': 'speedi', 
        'speedi grille': 'speedi', 
        'speedi products': 'speedi', 
        'speedi vent': 'speedi', 
        'pass and seymour': 'seymour',
        'pavestone rumblestone': 'rumblestone',
        'philips advance':'philips',
        'philips fastener':'philips',
        'philips ii plus':'philips',
        'philips manufacturing company':'philips',
        'safety first':'safety 1st',
        'sea gull lighting': 'sea gull',
        'scott':'scotts',
        'scotts earthgro':'scotts',
        'south shore furniture': 'south shore', 
        'tafco windows': 'tafco',
        'trafficmaster allure': 'trafficmaster', 
        'trafficmaster allure plus': 'trafficmaster', 
        'trafficmaster allure ultra': 'trafficmaster', 
        'trafficmaster ceramica': 'trafficmaster', 
        'trafficmaster interlock': 'trafficmaster', 
        'thomas lighting': 'thomas', 
        'unique home designs':'unique home',
        'veranda hp':'veranda',
        'whitehaus collection':'whitehaus',
        'woodgrain distritubtion':'woodgrain',
        'woodgrain millwork': 'woodgrain', 
        'woodford manufacturing company': 'woodford', 
        'wyndham collection':'wyndham',
        'yardgard select': 'yardgard',
        'yosemite home decor': 'yosemite'
        }

    def replace(self, text):
        for word in self.replace_brand_dict.keys():
            while word in text:
                text.replace(word, self.replace_brand_dict[word])
        return text
    
    
