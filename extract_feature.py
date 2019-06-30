# coding: utf-8                                                                                                                                                                                                                       # coding: utf-8
import sys
import codecs
import math
import copy
import re
import nltk
from log import *
from nltk.corpus import wordnet as wn
import xlrd
from nltk.collocations import *
from detect_cluster import Community
from util import *
reload(sys)
sys.setdefaultencoding('utf8')




##################################################################################
#从每个聚簇中，以tfidf值最大的一个关键字开始，以贪婪的方式提取特征
##################################################################################
def extractFeatures(sentences_list,community_list,community_keyword_list,all_words_list):
    community_feature_list=[]
    community_selected_sentence_id_list=[]
    j=0
    for community in community_list:
        #从社区的句子中抽取出来的词语搭配以及其频率
        features_dic=extractFeatureFromCommunity(community,sentences_list,community_keyword_list[j],all_words_list)
        #合并同义词词组，并选择一个支持度最大的
        selected_feature_set,selected_sentence_id=filterAndSelectFeatures(features_dic)
        community_feature_list.append(selected_feature_set)
        community_selected_sentence_id_list.append(selected_sentence_id)
        j+=1
    return community_feature_list,community_selected_sentence_id_list



##################################################################################
#统计一个社区的所有句子中的包含某个关键词的所有词组集合，以及他们的频率
##################################################################################  
def extractFeatureFromCommunity(community,sentence_list,keyword_list,all_words_list):
    features_dic={}
    #针对keyword_list里边的每个关键词，去抽取他们的词语搭配
    for keyword in keyword_list:
        #当前社区的成员
        menmbers_dic=community.getMenmberSentences()
        
        
        for (sentence_id, tfidf_vector) in menmbers_dic.items():
            #生成社区约简之后的句子
            left_words_list=[]
            for i in range(0,len(tfidf_vector)):
                if tfidf_vector[i]>0:
                    left_words_list.append(all_words_list[i])
                    
            #按照原有句子中单词的顺序产生约简之后的句子
            reduced_sentence=[]
            for word in sentence_list[sentence_id]:
                if word in left_words_list:
                    reduced_sentence.append(word)
                    
            #从一个句子中，根据关键词，产生包含关键词的所有长度为2的词组        
            #collocations_list,collocations_with_sort_list=getCollocationsFromSentence(sentence_list[sentence_id], keyword)
            #选择长度为2的词组
            collocations_list,collocations_with_sort_list=getCollocationsFromSentence_new(reduced_sentence, keyword)
            #collocations_list,collocations_with_sort_list=getCollocationsFromSentence_new(sentence_list[sentence_id], keyword)

            for i in range(0,len(collocations_list)):
                collocation_set=set(collocations_list[i])
                if collocation_set not in features_dic.keys():
                    item=[]
                    item.append(1)#在item中的位置为0,支持度
                    item.append(sentence_id)#在item中的位置为1，代表性句子的ID
                    item.append(collocations_with_sort_list[i])#在item中的位置为2
                    features_dic[frozenset(collocation_set)]=item
                else:
                    item=features_dic[frozenset(collocation_set)]
                    item[0]=item[0]+1
        
    return features_dic
                
                
        
##################################################################################
#过滤掉支持度相同，但是却是别人子集的特征词组，并选择支持度最大的一个特征作为特征
##################################################################################      
def filterAndSelectFeatures(features_dic): 
    for (feature,item) in features_dic.items():
        item.append(feature)#在item中的位置为3
        item.append(len(feature))#把长度添加进去,在item中的位置为4
        
    features_list=features_dic.values()
    #合并同义词,从后往前进行合并，如果是同义词，则将后者给删除，并合并二者的支持度
    num_features=len(features_list)
    i=num_features-1
    while(i>0):
        for j in range(0,i-1):
            #如果长度不相等，则直接continue
            if len(features_list[i][3])!=len(features_list[j][3]):
                continue      
            #如果找到同义词
            if isSynset_phrase(features_list[i][3], features_list[j][3])==True:
                #合并支持度
                features_list[j][0]=features_list[j][0]+features_list[i][0]
                #删除该同义词,并跳出循环
                features_list.pop(i)
                break
        i=i-1
        
    #将过滤之后的词语搭配按照支持度进行排序,如果支持度相等，则选择较长的一个
    new_sorted_feature_list=sorted(features_list, key=lambda p:(p[0],p[4]), reverse=True)
    
    #writelog("curr community features:**************")
    selected_feature=[]
    selected_feature_represent_sentence=[]
    size=len(new_sorted_feature_list)
    if size>3:
        size=3
    for i in range(0,size):
        #writelog(" ".join(new_sorted_feature_list[i][2])+" "+str(new_sorted_feature_list[i][0]))
        selected_feature.append(new_sorted_feature_list[i][2])
        selected_feature_represent_sentence.append(new_sorted_feature_list[i][1])
   
    #选择支持度最大的一个短语搭配作为特征并返回,并返回代表性句子的ID
    #return new_sorted_feature_list[0][2],new_sorted_feature_list[0][1]
    #这里是返回3个，而不是一个特征，这样能够更加准确
    return selected_feature,selected_feature_represent_sentence


            
##################################################################################
#从每句话中抽取包含keyword距离在前后5个单词范围内的所有长度为2的词组
##################################################################################
def getCollocationsFromSentence_new(sentence_list, keyword):
    #首先确定关键词在句子中的位置
    threshold_distance=5
    
    index_list=[]
    k=0
    for word in sentence_list:
        if word == keyword:
            index_list.append(k)
        k+=1
        
    all_feature_list=[]
    for index in index_list:
        #提取关键词前面的词所组成的特征
        star=index-threshold_distance+1
        if star<0:
            star=0
        for i in range(star,index):
            new_feature=[]
            new_feature.append(sentence_list[i])
            new_feature.append(keyword)
            all_feature_list.append(new_feature)
        #提取关键词后面的词所组成的特征
        end=index+threshold_distance-1
        if end>len(sentence_list):
            end=len(sentence_list)
        for i in range(index+1,end):
            new_feature=[]
            new_feature.append(keyword)
            new_feature.append(sentence_list[i])
            all_feature_list.append(new_feature)  
    result_feature_list,feature_with_sort_list=filterCollocations(all_feature_list)
    return result_feature_list,feature_with_sort_list        
    

##################################################################################
#从每句话中抽取长度为2的所有词组，并统计他们的词频
##################################################################################
def getCollocationsFromSentence_method_1_original(sentence):
    finder = BigramCollocationFinder.from_words(sentence,window_size = 5)
    bigram_dic=finder.ngram_fd
    #去除排序不同的词组，也即两个词组，如何单词相同，位置不同，则合并或者去除，并保留一个他们原始的顺序
    #其实，如果是连个单词构成的词组，保留他们的顺序也无意义，先这样吧
    result_feature_list,feature_with_sort_list=filterCollocations(bigram_dic.keys())
    return result_feature_list,feature_with_sort_list

##################################################################################
#从每句话中抽取包含keyword距离在前后5个单词范围内的所有词组
##################################################################################
def getCollocationsFromSentence(sentence_list, keyword):
    #首先确定关键词在句子中的位置
    threshold_distance=3
    
    index_list=[]
    k=0
    for word in sentence_list:
        if word == keyword:
            index_list.append(k)
        k+=1
        
    all_feature_list=[]
    for index in index_list:
        #提取关键词后面的词所组成的特征
        prefix=[]
        prefix.append(keyword)
        star=index+1
        end=index+threshold_distance
        if end>len(sentence_list):
            end=len(sentence_list)
        getFeatures(sentence_list,star,end,prefix,all_feature_list)
        #提取关键词前面的词所组成的特征
        first=index-threshold_distance+1
        if first<0:
            first=0
        for i in range(first,index):
            prefix=[]
            prefix.append(sentence_list[i])
            star=i+1
            end=index
            temp_feature_list=[]
            temp_feature_list.append(prefix)
            getFeatures(sentence_list,star,end,prefix,temp_feature_list)
            for feature in temp_feature_list:
                feature.append(keyword)
                all_feature_list.append(feature)
                #print " ".join(feature)
                star=index+1
                end=i+threshold_distance
                if end>len(sentence_list):
                    end=len(sentence_list)
                getFeatures(sentence_list,star,end,feature,all_feature_list)
                
    result_feature_list,feature_with_sort_list=filterCollocations(all_feature_list)
    return result_feature_list,feature_with_sort_list
                
            
def getFeatures(sentence_list,star,end,prefix,feature_list):
    if star>=end:
        return
    for i in range(star, end):
        new_feature=copy.deepcopy(prefix)
        new_feature.append(sentence_list[i])
        feature_list.append(new_feature)
        getFeatures(sentence_list,i+1,end,new_feature,feature_list)


def filterCollocations(all_feature_list):
    new_feature_list=[]
    feature_with_sort_list=[]
    for feature in all_feature_list:
        feature_set=set(feature)
        if feature_set not in new_feature_list:
            new_feature_list.append(feature_set)
            feature_with_sort_list.append(feature)
    return new_feature_list,feature_with_sort_list
    

 
##sentence_list=["all", "the", "words", "list", "of", "the", "sentences", "in", "the", "community"]
##all_feature_list=getCollocationsFromSentence(sentence_list, "of")
##for feature in all_feature_list:
##    print " ".join(feature)    
