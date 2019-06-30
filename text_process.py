# coding: utf-8                                                                                                                                                                                                                       # coding: utf-8
import sys
import codecs
import math
import copy
import re
import nltk
from log import *
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import xlrd

reload(sys)
sys.setdefaultencoding('utf8')




##################################################################################
#去停用词
##################################################################################
def deleteStopWords(stopwords,words_list):
    result = []
    for word in words_list:
        word = word.strip()
        word = word.lower()
        if word not in stopwords:
            result.append(word)
    return result


##################################################################################
#英文词词形还原
##################################################################################
def wordLemmatizer(wordnet_lemmatizer,words_list):
    result = []
    for word in words_list:
        word = word.strip()
        word = word.lower()
        word=wordnet_lemmatizer.lemmatize(word)
        result.append(word)
    return result


##################################################################################
#英文词词干化
##################################################################################
def stemWord(words_list):
    result = []
    for word in words_list:
        word = word.strip()
        word = word.lower()
        aa = wn.morphy(word)
        if aa==None:#我发现词干化会出现返回为空的情况
            result.append(word)
        else:
            result.append(aa)
    return result

##################################################################################
#分句
##################################################################################
# 设置分句的标志符号；可以根据实际需要进行修改
#cutlist = ".!！？;]()".decode('utf-8')
cutlist = ".!！？".decode('utf-8')

# 检查某字符是否分句标志符号的函数；如果是，返回True，否则返回False
def FindToken(cutlist, char):
    if char in cutlist:
        return True
    else:
        return False
    
# 进行分句的核心函数
def Cut(cutlist, lines):  #参数1：引用分句标志符；参数2：被分句的文本，为一行中文字符
    l = []  # 句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
    line = []  # 临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空
    p = 0;
    start = 0;
    for i in lines:  #对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）
        if FindToken(cutlist, i):  # 如果当前字符是分句符号
            if start == p:
                p += 1
                start = p
                continue
            # line.append(i)          #将此字符放入临时列表中
            l.append(''.join(line))  # 并把当前临时列表的内容加入到句子列表中
            start = p + 1
            line = []  # 将符号列表清空，以便下次分句使用
        else:  # 如果当前字符不是分句符号，则将该字符直接放入临时列表中
            line.append(i)
        p += 1;
    if start < len(lines):
        l.append(''.join(line))
    return l



def sentenceSplite(product_summary):
    #返回的句子列表
    sentences_list=[]
     
    # 如果是空行,则去掉空行
    if product_summary.count('\n')==len(product_summary):
        return
        
    sentences_from_summary=product_summary.split("\n")
    for sentence in sentences_from_summary:
        if sentence!="":
            #分句，因为summary里边有可能一段是多句话
            lines = Cut(list(cutlist), list(sentence.decode('utf-8', 'ignore')))
            for line in lines:
                line=line.strip()# 去掉首尾的空白
                if line=="" or line==None:
                    continue
                line = re.sub('[^A-Za-z]', ' ', line)
                line = line.strip()
                line = re.sub(r"\s{2,}", " ", line)
                sentences_list.append(line)
                    
    return sentences_list




##################################################################################
#对产品描述进行预处理,一个类别的产品形成一个词语list
##################################################################################
def summaryPreprocess(textPath):
    #product_dic = {}
    cluster_corpos=[]#一个类别的产品形成一个词语list
    sentences_list=[]#保存一个类别下的所有产品的描述语句
    original_sentences_list=[]#保存所有词的完整的产品描述的句子

     #从 excel表里边读取数据
    workbook = xlrd.open_workbook(textPath)
    sheet = workbook.sheet_by_index(0) # sheet索引从0开始
    n_row=sheet.nrows
    
    #为了调试程序使用
    """ """
    if n_row>20:
        n_row=20
    

    # 获取nltk的停用词表
    from nltk.corpus import stopwords
    stoplist = stopwords.words('english')


    #词性还原对象,这里发现词性还原的效果可能还没有词干化的效果好
    wordnet_lemmatizer = WordNetLemmatizer()

    for i in range(0, n_row):
        product_name=sheet.cell(i,0).value.encode('utf-8')
        product_summary=sheet.cell(i,1).value.encode('utf-8')
        
        #记录一个产品的所有句子
        product_sentences_list=[]
        
        #先分句
        original_product_sentences_list=sentenceSplite(product_summary)
        for sentence in original_product_sentences_list:
            #分词
            words_list = nltk.word_tokenize(sentence)
            #writelog(words_list)
        
            #词性判断，保留动词，名词，形容词
            words_list_tag = nltk.pos_tag(words_list)
            num_word=len(words_list_tag)
            keeped_words_list=[]
            for i in range(0,num_word):
                if words_list_tag[i][1] in ("NN","NNS","NNP","NNPS","VB","VBD","VBP","VBZ","VBG","VBN","JJ"):
                    keeped_words_list.append(words_list_tag[i][0])
            #writelog(keeped_words_list)
        
            #去停用词
            words_list_delstopword = deleteStopWords(stoplist, keeped_words_list)
            #writelog(words_list_delstopword)
        
            #词干化,词性还原的效果没有词干化的效果好，并且发现二者结合起来也没有多大改善，所以只用了词干化
            new_words_list=stemWord(words_list_delstopword)
            #new_words_list = wordLemmatizer(wordnet_lemmatizer,new_words_list_1)
            
            
            #进一步过滤只有一个字母的单词
            newer_words_list=[]
            for word in new_words_list:
                if len(word)>1:
                    newer_words_list.append(word)
                    
            #过滤去停用词之后，只有一个词或者为空的句子
            if len(newer_words_list)<=1:
                continue
            
            #去掉同一个产品内相同的句子,也即过滤无用词之后，剩下的词相同
            if isExist(newer_words_list,product_sentences_list)==True:
                continue
            
            #添加到当前产品描述的句子集合中
            product_sentences_list.append(newer_words_list)
            
            #将当前的句子和原始句子给记录下来
            cluster_corpos.extend(newer_words_list)
            sentences_list.append(newer_words_list)
            original_sentences_list.append(sentence)
            
    return cluster_corpos,sentences_list,original_sentences_list



def isExist(sentence_words_list,setences_list):
    for sentence in setences_list:
        str_sentence_a=" ".join(sentence)
        str_sentence_b=" ".join(sentence_words_list)
        if str_sentence_a==str_sentence_b:
            return True
    return False
        
    

##################################################################################
#读取一个目录下的所有文件
##################################################################################
def getAllFilePath(dirPath):
    import os
    import os.path

    files = os.listdir(dirPath)
    file_path_list=[]
    for file in files :
        #准确获取一个txt的位置，利用字符串的拼接
        txt_path = dirPath+"\\"+file
        file_path_list.append(txt_path)
        #print txt_path
    return file_path_list
    


def getALLFileContent(dirPath):
    file_path_list=getAllFilePath(dirPath)
    corpus=[]
    cluster_sentences_list=[]
    cluster_original_sentences_list=[]
    i=0
    for file_path in file_path_list:
        cluster_corpos,setences_list,original_sentences_list=summaryPreprocess(file_path)
        corpus.append(cluster_corpos)
        cluster_sentences_list.append(setences_list)
        cluster_original_sentences_list.append(original_sentences_list)
        if i==7:
            print str(i)+":"+file_path
        i+=1
    return corpus,cluster_sentences_list,cluster_original_sentences_list

        





