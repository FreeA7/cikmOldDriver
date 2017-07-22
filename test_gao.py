# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from lxml import etree
import multiprocessing
import datetime


# 从数据中获取标签行并返回为array
def getlabels(path):
    f = open(path)
    label = []
    while 1:
        line = f.readline()
        if line:
            label.append(int(re.sub('[^0-1]', '', line)))
        else:
            break
    return np.asarray(label)


# 去除列表中的所有空值
def clearlistnull(list_arg):
    while 1:
        try:
            list_arg.remove('')
        except:
            break
    return list_arg


# 重新组合被由属性名切开的列表，例如：
# 'Color:yellow Type:xxl Material:gold'
# --按照属性名切分--> ['Color:','yellow','Type:','xxl','Material:','gold']
# --recombinelist--> ['Color:yellow','Type:xxl','Material:gold']
def recombinelist(list_arg):
    list_output = []
    for i in range(len(list_arg)):
        if i % 2 == 1:
            list_output.append(list_arg[i - 1] + list_arg[i])
    return list_output


# 获取一行的短文本描述并做基础处理
def getshortdescription(data, i):
    html = str(data.get_value(i, 'short_description'))

    # 可能出现错误编码，对文本进行重编码
    html = html.encode().decode('unicode_escape')

    # 文本中存在big/small这种形式，即/并非描述之间的分隔而仅仅是or的意义
    # 因此会对之后的切分产生误判
    # 需要将所有的 big/small --> big small
    replacepattern = re.compile(' [a-z]+/[a-z]+ ')
    replaceword = replacepattern.findall(html)
    if replaceword:
        for word in replaceword:
            html = html.replace(word, word.replace('/', ' '))

    # html中特殊字符的替换
    html = html.replace(' :', ':')
    html = html.replace('&nbsp', ' ')
    html = html.replace('&amp', '&')
    html = html.replace('&quot', '"')
    html = html.replace('&lt', '<')
    html = html.replace('&gt', '>')
    html = re.sub(r"&#39;", "'", html)

    # 去除括号以及括号中的所有内容
    html = re.sub(r'[(][^)]+[)]', '', html)

    return html


# 判断短描述
def judgeattr(attrlist, html):
    for attr in attrlist:
        if attr in html:
            return 1
    return 0


# 解析短描述文本，将一段话分割为多句短描述并存放在一个list中
def parserdes(q_info, q_error, data, alphabet, number, attrpattern, attrlist):
    try:
        for idx in data.index:

            # 获取一段话准备切割
            html = getshortdescription(data, idx)

            # 短描述为空
            if html in ['……', '…', 'nan', '.']:
                descs = []

            # 使用ul/ol标签以及li标签格式
            elif '</ul>' in html or '</ol>' in html or '</li>' in html:
                html = etree.HTML(html)
                descs = html.xpath('//li')
                descs = [desc.text for desc in descs]

            # 使用tr标签以及td标签格式
            elif '</tr>' in html:
                html = etree.HTML(html)
                descs = html.xpath('//td')
                descs = [desc.text for desc in descs]

            # 使用br标签分隔
            elif '<br' in html:
                descs = re.split('<br[^>]*>', html)
                descs = clearlistnull(descs)

            # 使用p标签
            elif '<p' in html:
                temp_pattern = re.compile('<p[^>]*>[^<]*</p')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>') + 1:-3] for desc in descs]
                descs = clearlistnull(descs)

            # 使用h2标签
            elif '<h2' in html:
                temp_pattern = re.compile('<h2[^>]*>[^<]*</h2')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>') + 1:-4] for desc in descs]
                descs = clearlistnull(descs)

            # 使用b标签
            elif '<b' in html:
                temp_pattern = re.compile('<b[^>]*>[^<]*</b')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>') + 1:-3] for desc in descs]
                descs = clearlistnull(descs)

            # 使用div标签
            elif '<div' in html:
                temp_pattern = re.compile('<div[^>]*>[^<]*</div')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>') + 1:-5] for desc in descs]
                descs = clearlistnull(descs)

            # 使用/分隔，但注意这种情况说明一定没有html标签，所以/一定不是</中的/
            elif '/' in html and '</' not in html:
                descs = html.split('/')
                descs = clearlistnull(descs)

            # 不同短描述是不同的属性，以属性+：的形式开头
            elif judgeattr(attrlist, html):
                descs = re.split(attrpattern, html)
                descs = clearlistnull(descs)
                if len(descs) % 2 == 1:
                    descs = descs[1:]
                descs = recombinelist(descs)

            # 使用1,2,的形式分隔
            elif '1,' in html and '2,' in html and '3,' in html:
                descs = re.split('[0-9]+,', html)
                descs = clearlistnull(descs)

            # 以上分割形式都没能处理
            else:
                descs = []

                # bigSmall这种形式，Small就是另一个短描述的开头
                for word in html[1:]:
                    if word in alphabet and html[html.find(word) - 1] not in alphabet + ' -()' + number:
                        descs.append(html[:html.find(word)])
                        html = html[html.find(word):]
                descs.append(html)

                # 依然没有被分开的话使用•;.进行最后的分隔处理
                if len(descs) == 1:
                    descs = descs[0].split('•')
                if len(descs) == 1:
                    descs = descs[0].split(';')
                if len(descs) == 1:
                    descs = descs[0].split('.')
                descs = clearlistnull(descs)

            # 将短描述的list通过一个quene传回主程序
            id = [str(idx)]
            id.extend(descs)
            q_info.put(id)
            #result = np.array([idx, descs])
            #q_info.put(result)

    # 如果处理的过程报错，则将错误信息通过quene传回打印
    except Exception as err:
        q_error.put(err)

# 主函数
if __name__ == '__main__':

    # 定义进程池多进程同时进行预处理
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()

    # 定义数据传输quene以及错误信息传输却呢
    q_info = manager.Queue(100)
    q_error = manager.Queue(100)

    # 定义属性名，便于之后的切割
    attrlist = ['Color:', 'Type:', 'Material:', 'Brand:', 'Gender:',
                'Size:', 'Weight:', 'Capacity:', 'Transfer rate:',
                'Interface:', 'Compatibility:', 'Model:', 'Length:',
                'Operate voltage:', 'Rate current:', 'Detecting distance:',
                'Colors available:', 'Pattern:', 'Style:', 'Design:', 'Name:',
                'type:', 'length:', 'Season:', 'Package Content:', 'ItemType:',
                'TopsType:', 'PatternType:', 'SleeveStyle:', 'FabricType:',
                'Hooded:', 'Collar:', 'SleeveLength:', 'Interior:', 'LiningMaterial:',
                'Shape:', 'Occasion:', 'HandbagsType:', 'Hardness:', 'ClosureType:',
                'Straps:', 'BrandName:', 'MainMaterial:', 'Exterior:', 'Decoration:',
                'Movement:', 'Features:', 'Fashion:', 'ModelNumber:', 'Typesofbags:',
                'Printed:', 'Use:', 'ApplyTo:', 'Hooded:', 'ClothingLength:', 'Instock:',
                'Shipping:', 'ColorStyle:', 'scene:', 'fabric:', 'Description:',
                'Description1', 'GenuineLeatherType:', 'Wallets:', 'WalletLength:',
                'ItemHeight:', 'ItemLength:', 'ItemWeight:', 'ItemWidth:',
                'MaterialComposition:', 'is_customized:']

    # 由于属性名在文本中可能首字母大写也可能不大写，因此都考虑进来
    attrlistlower = [j.lower() for j in attrlist]
    attrlist += attrlistlower
    attrpattern = '(' + '|'.join(attrlist) + ')'

    # 定义大写字母字母表以及数字
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    number = '0123456789'

    # 打开训练文件并获取两个labels并与数据结合
    data = pd.read_csv('./data/training/data_train.csv', encoding='utf-8',
                       names=['country', 'sku_id', 'title', 'category_lvl_1',
                              'category_lvl_2', 'category_lvl_3',
                              'short_description', 'price', 'product_type'])
    clarity = getlabels('./data/training/clarity_train.labels')
    conciseness = getlabels('./data/training/conciseness_train.labels')
    data['clarity'] = clarity
    data['conciseness'] = conciseness

    # 将数据分为workers份
    data_list = []
    workers = 8

    # 创建进程进行多进程处理
    for i in range(workers):
        data_list.append(data.sample(frac=(1 / (workers - i))))
        data.drop(data_list[-1].index)
        print('添加第' + str(i) + '个进程 : ' + str(datetime.datetime.now()))
        pool.apply_async(parserdes, args=(q_info, q_error, data_list[i], alphabet, number, attrpattern, attrlist))
        print('第' + str(i) + '个进程添加完毕 : ' + str(datetime.datetime.now()))
    pool.close()

    # results = np.empty([1, 2])

    # 输出处理之后的结果
    count = 0
    while not q_info.empty():
        count += 1
        get_one_row = q_info.get(True)
        # print(get_one_row)
        #results = np.concatenate((results, np.array(get_one_row)), axis=0)
        # print("added" + str(count))
        # print(get_one_row)
        fw = open('output.txt', 'a+', encoding='utf-8')
        for elements in get_one_row:
            if elements:
                fw.write(elements+'\t')
        fw.write('\n')
        if not q_error.empty():
            print(q_error.get(True))
            # pass
        # if count % 1000 == 0:
        #     print(str(count // 1000) + 'k')
        #     print(str(len(data_list)))
            # pass

    print('数据处理完成，共 %d 条' % count)
