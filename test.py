import pandas as pd
import numpy as np
import re
from lxml import etree
import multiprocessing
import datetime

def culdata():
    f = open('./data/training/data_train.csv','rb')
    count = 0
    countall = 0
    while 1:
        line = f.readline()
        if line:
            countall += 1
            try:
                line.decode('utf-8')
            except:
                count += 1
        else:
            break
    print('countall : %d'%countall)
    print('count : %d'%count)

def getlabels(path):
    f = open(path)
    label = []
    while 1:
        line = f.readline()
        if line:
            label.append(int(re.sub('[^0-1]','',line)))
        else:
            break
    return np.asarray(label)

def clearlistnull(list_arg):
    while 1:
        try:
            list_arg.remove('')
        except:
            break
    return list_arg

def recombinelist(list_arg):
    list_output = []
    for i in range(len(list_arg)):
        if i%2 == 1:
            list_output.append(list_arg[i-1] + list_arg[i])
    return list_output

def getshortdescription(data,i):
    html = str(data.get_value(i, 'short_description'))

    html = html.encode().decode('unicode_escape')
##    html = html.replace('USB',' usb ')
##    html = html.replace('LED',' led ')
##    html = html.replace('TPU',' tpu ')
##    html = html.replace('CPU',' cpu ')
##    html = html.replace('GPU',' gpu ')
##    html = html.replace('LCD',' lcd ')
##    html = html.replace('CD',' cd ')
##    html = html.replace('HDD',' hdd ')
##    html = html.replace('Xbox',' xbox ')
##    html = html.replace('DVD',' dvd ')
##    html = html.replace('PC',' personal computer  ')
    
    replacepattern = re.compile(' [a-z]+/[a-z]+ ')
    replaceword = replacepattern.findall(html)
    if replaceword:
        for word in replaceword:
            html = html.replace(word, word.replace('/',' '))
            
    html = html.replace(' :', ':')
    html = html.replace('&nbsp', ' ')
    html = html.replace('&amp', '&')
    html = html.replace('&quot', '"')
    html = html.replace('&lt', '<')
    html = html.replace('&gt', '>')
##    html = html.replace(html[0], html[0].lower(), 1)
        
    return html

def parserdes(q_info, q_error, data, alphabet, number, attrpattern):
    try:
    ##    for i in range(9462 + 1, len(data)):
        for idx in data.index:
            html = getshortdescription(data, idx)
            if html in ['……','…','nan','.']:
                descs = []
            elif '</ul>' in html or '</ol>' in html or '</li>' in html:
                html = etree.HTML(html)
                descs = html.xpath('//li')
                descs = [desc.text for desc in descs]
            elif '</tr>' in html:
                html = etree.HTML(html)
                descs = html.xpath('//td')
                descs = [desc.text for desc in descs]
            elif '<br' in html:
                descs = re.split('<br[^>]*>',html)
                descs = clearlistnull(descs)
            elif '<p' in html:
                temp_pattern = re.compile('<p[^>]*>[^<]*</p')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>')+1:-3] for desc in descs]
                descs = clearlistnull(descs)
            elif '<h2' in html:
                temp_pattern = re.compile('<h2[^>]*>[^<]*</h2')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>')+1:-4] for desc in descs]
                descs = clearlistnull(descs)
            elif '<b' in html:
                temp_pattern = re.compile('<b[^>]*>[^<]*</b')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>')+1:-3] for desc in descs]
                descs = clearlistnull(descs)
            elif '<div' in html:
                temp_pattern = re.compile('<div[^>]*>[^<]*</div')
                descs = temp_pattern.findall(html)
                descs = [desc[desc.find('>')+1:-5] for desc in descs]
                descs = clearlistnull(descs)
            elif '/' in html and '</' not in html:
                descs = html.split('/')
                descs = clearlistnull(descs)
            elif 'Capacity:' in html or 'Color:' in html or 'Operate voltage:' in html or 'Material:' in html or 'Size:' in html or 'Type:' in html:
                descs = re.split(attrpattern,html)
                descs = clearlistnull(descs)
                if len(descs)%2 == 1:
                    descs = descs[1:]
                descs = recombinelist(descs)
            elif '1,' in html and '2,' in html and '3,' in html:
                descs = re.split('[0-9]+,',html)
                descs = clearlistnull(descs)
            else:
                descs = []
                for word in html[1:]:
                    if word in alphabet and html[html.find(word)-1] not in alphabet+' -()'+number:
                        descs.append(html[:html.find(word)])
                        html = html[html.find(word):]
                descs.append(html)
                if len(descs) == 1:
                    descs = descs[0].split('•')
                if len(descs) == 1:
                    descs = descs[0].split(';')
                if len(descs) == 1:
                    descs = descs[0].split('.')
                descs = clearlistnull(descs)

                    
            descs = [re.sub('[(][^)]+[)]','',desc) for desc in descs]
                      
            q_info.put(descs)
    except Exception as err:
        q_error.put(err)




if __name__ == '__main__':

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    q_info = manager.Queue(100)
    q_error = manager.Queue(100)

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
    attrlistlower = [j.lower() for j in attrlist]
    attrlist += attrlistlower
    attrpattern = '('+'|'.join(attrlist)+')'

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    number = '0123456789'
                
    data = pd.read_csv('./data/training/data_train.csv',encoding = 'utf-8',
                       names = ['country', 'sku_id', 'title', 'category_lvl_1',
                                'category_lvl_2', 'category_lvl_3',
                                'short_description', 'price', 'product_type'])
    clarity = getlabels('./data/training/clarity_train.labels')
    conciseness = getlabels('./data/training/conciseness_train.labels')
    data['clarity'] = clarity
    data['conciseness'] = conciseness

    data_list = []
    workers = 8

    for i in range(workers):
        data_list.append(data.sample(frac=(1/(workers-i))))
        data.drop(data_list[-1].index)
        print('添加第'+str(i)+'个进程 : ' + str(datetime.datetime.now()))
        pool.apply_async(parserdes, args=(q_info, q_error, data_list[i], alphabet, number, attrpattern))
        print('第'+str(i)+'个进程添加完毕 : ' + str(datetime.datetime.now()))
    pool.close()

    count = 0
    while 1:
        count += 1
        print(q_info.get(True))
        if not q_error.empty():
            print(q_error.get(True))
        if count % 100==0:
            print(str(count//100)+'h')
            
            
