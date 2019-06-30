# coding: utf-8                                                                                                                                                                                                                       # coding: utf-8
import sys
import codecs
import math
import copy
import re
import nltk
from log import *
import xlrd
from cluster import Community

reload(sys)
sys.setdefaultencoding('utf8')
"""
这个文件中的方法是，根据一种规则选出K个社区,每个社区的质心以及成员向量均在该社区对象内
"""

##################################################################################
#对获取的各个社区进行排序，过滤噪声社区，提取k个社区
##################################################################################
#根据质心的权重以及社区的紧密程度进行排序
def selectTopKCommunity(community_list,all_words_list,k,WEIGHT_COEFFICIENT):
    result_community_list=[]
    result_keyword_list=[]

    size=len(community_list)
    if size>k:
        size=k

    #排序并选择
    selected_community_id_list=sortandSelectCommunity(community_list,size,WEIGHT_COEFFICIENT)
    #保存结果
    for i in range(0,size):
        result_community_list.append(community_list[selected_community_id_list[i]])
        #获取当前社区的topk的关键词
        keyword_list=community_list[selected_community_id_list[i]].getDominateWords()
        #将关键词的标号转换成真正的词
        words_list=[]
        for keyword_id in keyword_list:
            words_list.append(all_words_list[keyword_id])
        result_keyword_list.append(words_list)
        
    return result_community_list,result_keyword_list


##################################################################################
#对各个社区进行排序
##################################################################################
def sortandSelectCommunity(community_list,size,WEIGHT_COEFFICIENT):
    #计算各个社区的紧密程度，也就是社区各个成员与其质心的平均距离
    from util import caculatCosine
    distance_list=[]
    centroid_score_list=[]
    size_list=[]
    for i in range(0,len(community_list)):
        #质心
        centroid=community_list[i].getCentroid()
        #成员
        menmbers_dic=community_list[i].getMenmberSentences()
        #计算平均相似度
        total_distance=0
        for sentence_id in menmbers_dic.keys():
            distance=caculatCosine(menmbers_dic[sentence_id],centroid)
            total_distance+=distance
        average_distance=total_distance/float(len(menmbers_dic))
        distance_list.append(average_distance)
        
        #计算质心的权重
        keywords_list=community_list[i].getDominateWords()
        centroid_score=getTopkKeywordAverageWeight(centroid,keywords_list)
        centroid_score_list.append(centroid_score)

        #计算社区大小
        size_list.append(len(menmbers_dic))


    #将社区紧密程度的权重和质心权重进行归一化
    max_distance=0
    for distance in distance_list:
        if distance>max_distance:
            max_distance=distance
    for i in range(0,len(distance_list)):
        distance_list[i]=distance_list[i]/max_distance

    max_score=0
    for score in centroid_score_list:
        if score > max_score:
            max_score=score
    for i in range(0,len(centroid_score_list)):
        centroid_score_list[i]=centroid_score_list[i]/max_score

    max_size=0
    for size_item in size_list:
        if size_item>max_size:
            max_size=size_item
    for i in range(0,len(size_list)):
        size_list[i]=size_list[i]/max_size
    
        
        

    #通过质心权重与社区紧密程度的权重加权求和，计算每个社区的权重,并进行排序
    community_score_list=[[] for i in range(0,len(community_list))]
    """
    for i in range(0,len(community_list)):
        score_i=coefficient*distance_list[i]+(1-coefficient)*centroid_score_list[i]
        community_score_list[i].append(i)
        community_score_list[i].append(score_i)
    """
    for i in range(0,len(community_list)):
        score_i=WEIGHT_COEFFICIENT*size_list[i]+(1-WEIGHT_COEFFICIENT)*centroid_score_list[i]
        community_score_list[i].append(i)
        community_score_list[i].append(score_i)
        
    #排序
    sorted_result=sorted(community_score_list, key=lambda p: p[1], reverse=True)
    print "liuchun is here ", len(sorted_result)

    #返回当前权重最大的size个社区的id
    selected_community_id_list=[]
    for i in range(0,size):
        selected_community_id_list.append(sorted_result[i][0])
        
    return selected_community_id_list

 
    
def getTopkKeywordAverageWeight(centroid,keyword_list):
    num=0
    average_weight=0
    for keyword_id in keyword_list:
        if centroid[keyword_id]!=0:
            average_weight+=centroid[keyword_id]
            num+=1
            
    if num!=0:
        return average_weight/num
    else:
        return 0
     
    

        
            
            
            
    