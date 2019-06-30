# coding: utf-8
import sys
import math
import re
from log import *
from sklearn.cluster import AffinityPropagation
import nltk
from nltk.corpus import wordnet as wn
from nltk.collocations import *
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')



###################################################
# 得到所有的词
###################################################
def getAllWordsList(all_documents_list):
    all_words_list=[]
    for document in all_documents_list:
        for word in document:
            if word not in all_words_list:
                all_words_list.append(word)
    return all_words_list


###################################################
# 得到一个文档中各个词的频率
###################################################
def computeDocumentWordTF(document_words_list):
    document_word_count_dic={}
    for word in document_words_list:
        if word in document_word_count_dic.keys():
            document_word_count_dic[word]+=1
        else:
            document_word_count_dic[word]=1
        
        
    #进行归一化处理
    total_count=0
    for (word,count) in document_word_count_dic.items():
        total_count+=document_word_count_dic[word]
    for word in document_word_count_dic.keys():
        document_word_count_dic[word]=float(document_word_count_dic[word])/float(total_count)
    return document_word_count_dic
        
        
###################################################
# 得到各个词的idf值
###################################################
def computeIDF(all_words_list,all_documents_list):
    all_word_idf_dic={}
    num_document=len(all_documents_list)
    
    for word in all_words_list:
        all_word_idf_dic[word]=0
    for word in all_words_list:
        num_supp_document=0
        for document in all_documents_list:
            if word in document:
                num_supp_document+=1
                
        word_idf = math.log(float(num_document+1) / float(num_supp_document+1), 2)
        word_idf+=1
        all_word_idf_dic[word]=word_idf
    return all_word_idf_dic


###################################################
# 得到一个文档中词的tfidf值
###################################################       
def computeDocumentWordTFIDF(document_words_list,all_word_idf_dic):
    document_word_count_dic=computeDocumentWordTF(document_words_list)
    
    document_word_tfidf_dic={}
    for (word,tf) in document_word_count_dic.items():
        document_word_tfidf_dic[word]=document_word_count_dic[word]*all_word_idf_dic[word]
        
    return document_word_tfidf_dic
    
    
                

###################################################
#将一个文档转成TF-IDF向量
################################################### 
def documentVectorizer(all_words_list,document_words_list,all_word_idf_dic):
    document_word_count_dic=computeDocumentWordTF(document_words_list)
    
    document_vector_list=[]
    for word in all_words_list:
        curr_tfidf=0.0
        if word in document_word_count_dic.keys():
            curr_tfidf=document_word_count_dic[word]*all_word_idf_dic[word]
        
        document_vector_list.append(curr_tfidf)
    return document_vector_list



##################################################################################
#计算一个选定的类别的描述中，每个词的tfidf值
##################################################################################
def computeSelectedDocumentWordTFIDF(corpus,cluster_id):
    #先计算选定类别所有的词的集合
    all_words_list=[]
    for word in corpus[cluster_id]:
        if word not in all_words_list:
            all_words_list.append(word)
            
    #计算选定类别中每个词的idf值
    all_word_idf_dic=computeIDF(all_words_list,corpus)
    #在计算选定类别文档中，每个词的tfidf值
    cluster_word_tfidf_dic=computeDocumentWordTFIDF(corpus[cluster_id],all_word_idf_dic)
   
    return cluster_word_tfidf_dic

###################################################
#将多个文档转成TF-IDF向量矩阵
################################################### 
def documentListVectorizer(all_documents_list):
    #先计算所有的词的集合
    all_words_list=getAllWordsList(all_documents_list)
    #计算每个词的idf值
    all_word_idf_dic=computeIDF(all_words_list,all_documents_list)
    #定义一个数组，将每个文档转成一个向量
    import numpy as np
    num_documents=len(all_documents_list)
    num_words=len(all_words_list)
    documents_vector_array=np.zeros((num_documents,num_words))
    for i in range(0,num_documents):
        document_word_count_dic=computeDocumentWordTF(all_documents_list[i])
        for j in range(0,num_words):
            word_tfidf=0
            word=all_words_list[j]
            if word in document_word_count_dic.keys():
                word_tfidf=document_word_count_dic[word]*all_word_idf_dic[word]
            documents_vector_array[i,j]=word_tfidf
    return documents_vector_array

###################################################
#在给定每个词的tfidf值得情况下，将多个文档转成TF-IDF向量矩阵
###################################################
def documentsVectorizer_array(all_documents_list,word_tfidf_dic):
    all_words_list=word_tfidf_dic.keys()
    #定义一个数组，将每个文档转成一个向量
    num_documents=len(all_documents_list)
    num_words=len(all_words_list)
    documents_vector_array=np.zeros((num_documents,num_words))
    for i in range(0,num_documents):
        for j in range(0,num_words):
            word_tfidf=0
            word=all_words_list[j]
            if word in all_documents_list[i]:
                word_tfidf=word_tfidf_dic[word]
            documents_vector_array[i,j]=word_tfidf
            
    #将tfidf向量归一化       
    from sklearn.preprocessing import normalize
    new_document_vector_array=normalize(documents_vector_array, norm='l2')
    
    return new_document_vector_array,all_words_list


def documentsVectorizer_tfidf_list(all_documents_list,word_tfidf_dic):
    all_words_list=word_tfidf_dic.keys()
    #定义一个数组，将每个文档转成一个向量
    result_vector_list=[]
    num_documents=len(all_documents_list)
    num_words=len(all_words_list)
    for i in range(0,num_documents):
        item_vector=[]
        for j in range(0,num_words):
            word_tfidf=0
            word=all_words_list[j]
            if word in all_documents_list[i]:
                word_tfidf=word_tfidf_dic[word]
            item_vector.append(word_tfidf)
        result_vector_list.append(item_vector)
        
    return result_vector_list,all_words_list



############################################################
# 计算两个向量之间的余弦相似度
############################################################
def caculatCosine(vector_1, vector_2):
    """
    size=len(vector_1)
    size_2=len(vector_2)
    if size_2<=0:
        pass
    a=0
    b=0
    c=0
    for i in range(0, size):
        a+=vector_1[i]*vector_2[i]
        b+=vector_1[i]*vector_1[i]
        c+=vector_2[i]*vector_2[i]
        
    if b==0 or c==0:
        return 0
    
    b=math.sqrt(b)
    c=math.sqrt(c)
    return a/(b*c)
    """
    #当两个向量归一化的情况下，返回他们的余弦相识度
    size=len(vector_1)
    a=0
    for i in range(0, size):
        a+=vector_1[i]*vector_2[i]
    return a
    
  
##################################################################################
#计算聚簇的质心
##################################################################################
def getCentroid(tfidf_matrixt,coms):
    sum_vector=[]
    for sentence_id in coms:
        vectorAdd(sum_vector, tfidf_matrixt[sentence_id])
        
    #社区的大小    
    num=len(coms)
    size=len(sum_vector)
    for i in range(0, size):
        sum_vector[i] = sum_vector[i] /num
    return sum_vector
        
    
def vectorAdd(sum_vector, vector):
    size = len(sum_vector)
    # 如果是第一次进行一个类中的向量相加
    if size == 0:
        new_size = len(vector)
        for i in range(0, new_size):
            sum_vector.append(vector[i])    
    else:
        for i in range(0, size):
            sum_vector[i] = sum_vector[i] + vector[i]



def recomputeCentroid(tfidf_matrixt,centroid, new_member_id, comm_size):
    length=len(centroid)
    new_centroid=[]
    for i in range(0,length):
        centroid[i]=(centroid[i]/comm_size)*(comm_size/(comm_size+1))+tfidf_matrixt[new_member_id][i]/(comm_size+1)
        
        
    
       
##################################################################################
#按照AP聚类的方式，从聚簇的质心中找出该聚簇关键词的位置
#这里的假设是，聚簇的关键词往往在质心向量中的数值比较大
##################################################################################
def getKeyWords(centroid):
    result=[]
    
    #找出质心中所包含的所有词
    left_words_weight_list=[]
    for i in range(0,len(centroid)):
        if centroid[i]>0:
            item=[]
            item.append(i)
            item.append(centroid[i])
            left_words_weight_list.append(item)
            
    #print "left words in centroid,", left_words_weight_list
    
    #只有一个关键词，则直接返回该关键词就行
    if len(left_words_weight_list)<2:
        result.append(left_words_weight_list[0][0])
        #print "The keywords id are: ",result
        return result
        
    """        
    #计算剩余词之间的相识度
    num_words=len(left_words_weight_list)
    sim_matrix=np.zeros((num_words,num_words))
    #中位数
    average=0
    for i in range(0,num_words):
        for j in range(0,num_words):
            sim=abs(left_words_weight_list[i][1]-left_words_weight_list[j][1])
            sim_matrix[i][j]=sim
            #if sim>0 and sim>average:
            #    average=sim
            average+=sim
    average=average/float(num_words*num_words)
            
    #AP聚类
    af = AffinityPropagation(preference=average, affinity="precomputed").fit(sim_matrix)
    """

    #kmeans聚类
    from sklearn.cluster import KMeans
    data=np.array([item[1] for item in left_words_weight_list]).reshape(-1,1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    centroid_list=kmeans.cluster_centers_
    labels=kmeans.labels_
    n_clusters_ = len(centroid_list)
    #print "cluster centroids:",centroid_list
    print labels
    max_centroid=0
    max_cluster_id=0
    cluster_menmbers_list=[]
    for i in range(0, n_clusters_):
        """
        center_indice=cluster_centers_indices[i]
        if max_centroid<left_words_weight_list[center_indice][1]:
            max_centroid=left_words_weight_list[center_indice][1]
            max_cluster_id=i
        """
        centroid=centroid_list[i]
        if max_centroid<centroid:
            max_centroid=centroid
            max_cluster_id=i
        
        menmbers_list=[]    
        for j in range(0, len(labels)):
            if labels[j]==i:
                menmbers_list.append(j)
        cluster_menmbers_list.append(menmbers_list)
        
    for id in cluster_menmbers_list[max_cluster_id]:
        #需要返回的是原始质心中关键词的位置
        result.append(left_words_weight_list[id][0])
    
        
    #print "The keywords id are: ",result
    #返回质心最大的一个聚簇的成员列表
    return result


   
def getTopKKeyword(centroid,k):
    keyword_list=[]
    for i in range(0,k):
            max_weight=0
            max_weight_key=0
            for j in range(0,len(centroid)):
                if j not in keyword_list and centroid[j]>max_weight:
                    max_weight=centroid[j]
                    max_weight_key=j
            keyword_list.append(max_weight_key)
            
    return keyword_list
        

##################################################################################
#判断两个社区是否相同
##################################################################################
def isSame(list_a, list_b):
    if len(list_a)!=len(list_b):
        return False
    
    for item_a in list_a:
        if item_a not in list_b:
            return False
    return True


##################################################################################
#根据同义词
##################################################################################     
def getSynset(word):
    result=[]
    for synset in wn.synsets(word):
        synset_words=synset.lemma_names()
        for word in synset_words:
            if word not in result:
                result.append(word)
    return result

def isSynset_word(word_a, word_b):
    synset_a=getSynset(word_a)
    synset_b=getSynset(word_b)
    if (word_a in synset_b) or (word_b in synset_a):
        return True
    else:
        return False


#如果两个集合长度相同，并且集合中的每个元素都对应有同义词，则二者是同义的
#如果长度不相等，则不是同义词
#如果其中有一个词，在另一个集合中找不到同义词，则二者也不是同义的
def isSynset_phrase(set_a, set_b):
    if len(set_a)!=len(set_b):
        return False
    
    set_intersection=set_a & set_b
    set_reduce_a=set_a-set_intersection
    set_reduce_b=set(set_b-set_intersection)
 
    for word_a in set_reduce_a:
        Tag=False
        for word_b in set_reduce_b:
            if isSynset_word(word_a,word_b)==True:
                Tag=True
                set_reduce_b.remove(word_b)
                break
        if Tag==False:
            return False
        
    return True

    
