#coding=utf-8
from parameters import *
from log import *
import copy
from util import *
import numpy as np
from cluster import Community
import sys
sys.setrecursionlimit(1000000)



##################################################################################
#创建相识度矩阵，并在发现一个社区之后，更新相识性网络
##################################################################################        
def getAdjacentMatrixForSentences_tfidfvector(tfidf_matrixt):
    #先获取所有的单词的列表
    num_sentences=len(tfidf_matrixt)
    Adjacent_matrix= np.zeros((num_sentences,num_sentences))

    for i in range(0, num_sentences):
        for j in range(i+1, num_sentences):
            sim_i_j=caculatCosine(tfidf_matrixt[i],tfidf_matrixt[j])
            Adjacent_matrix[i][j]=sim_i_j
            Adjacent_matrix[j][i]=sim_i_j
        Adjacent_matrix[i][i]=0
            
    return Adjacent_matrix

def updateAdjacentMatrix(AdjacentMatrix,tfidf_matrixt,left_nodes,node_to_update):
    #只更新剩余节点之间的相识度
    for sentence_id_a in node_to_update:
        for sentence_id_b in left_nodes:
            sim_i_j=caculatCosine(tfidf_matrixt[sentence_id_a],tfidf_matrixt[sentence_id_b])
            AdjacentMatrix[sentence_id_a][sentence_id_b]=sim_i_j
            AdjacentMatrix[sentence_id_b][sentence_id_a]=sim_i_j
        AdjacentMatrix[sentence_id_a][sentence_id_a]=0
            
                      
##################################################################################
#寻找种子节点
##################################################################################
#从剩余节点中跟具有最大相似度的边相连的节点作为种子节点
def getSeedByGravity(AdjacentMatrix,left_nodes,MIN_SIMILARITY):
    max_gravity=0
    seed=-1
    for node_id in left_nodes:
        for i in range(0,len(AdjacentMatrix)):
            if i in left_nodes and max_gravity<AdjacentMatrix[node_id][i]:
                max_gravity=AdjacentMatrix[node_id][i]
                seed=node_id  
    if max_gravity>MIN_SIMILARITY and seed!=-1:
        return seed
    else:
        return -1
        

##################################################################################
#根据句子相似性网络，依据软分类算法，挖掘句子社区
##################################################################################
#获取自然子图结构，重叠节点
def getCommunity_BySentences(AdjacentMatrix,tfidf_matrixt,MIN_SIMILARITY,MAX_SIMILARITY):
    num_nodes=len(tfidf_matrixt)
    
    left_nodes = [i for i in range(num_nodes)]
    #tag_list=[0 for i in range(num_nodes)]
    graph = {}
    gnums = 0
    while(1):
        print "num_left_nodes:{}".format(len(left_nodes))
        new_community=set()
        #选择具有最大的相识度的边的节点为种子节点
        seed=getSeedByGravity(AdjacentMatrix,left_nodes,MIN_SIMILARITY)
        if seed !=-1:
            new_community.add(seed)
            centroid=tfidf_matrixt[seed]
            
            print 'before:','graphid',gnums,',seed',list(new_community)
            removed_nodes=[]
            Propagate(AdjacentMatrix,tfidf_matrixt,new_community,centroid,left_nodes,removed_nodes,MIN_SIMILARITY)
            print 'after:',list(new_community)

            
            #删除只有一个节点的社区
            if len(new_community)==1:
                writelog("a community with only one menmber is producted")
                #continue
            
            community_exis=0
            for (key, community) in graph.items():
                if isSame(set(community.getMenmberSentences().keys()),new_community)==True:
                    community_exis=1
                    writelog("The new community already exist!!")
                    print "The new community already exist!!"
                    #break
                    
           #创建一个新的社区
            menmber_sentence_dic={}
            for menmber_id in new_community:
                menmber_sentence_dic[menmber_id]=copy.deepcopy(tfidf_matrixt[menmber_id])
            #计算质心
            centroid=getCentroid(tfidf_matrixt,new_community)
            community_object=Community(centroid,menmber_sentence_dic)
            graph[gnums]=community_object
            #删除新社区中成员的关键词，并将边界成员释放掉，也就是可以作为新的种子来进行选择
            node_to_update=removeCommunityNodes(tfidf_matrixt,community_object,left_nodes,MAX_SIMILARITY)
            #更新相识性网络
            updateAdjacentMatrix(AdjacentMatrix,tfidf_matrixt,left_nodes,node_to_update)
                
            gnums += 1
                
        else:
            return graph
        


#迭代过程
def Propagate(AdjacentMatrix,tfidf_matrixt,coms,centroid,left_nodes,removed_nodes,MIN_SIMILARITY):
    #获取当前社区的所有邻居节点以及邻居节点到当前社区质心的距离
    nel, nelf = getCommunityNeighbours(AdjacentMatrix,tfidf_matrixt,coms,centroid,left_nodes,removed_nodes)
    #所有的邻居节点与质心的距离都小于阈值
    if len(nelf)>0 and max(nelf) >= MIN_SIMILARITY:
        #将当前距离质心最近的一个节点加入到社区中
        t = nel[nelf.index(max(nelf))]
        
      
        coms.add(t)
        print "new node is added: {}".format(t)
        #重新计算质心
        centroid=getCentroid(tfidf_matrixt,coms)
        
        #去掉离质心距离小于阈值的所有节点
        #removeCommunityMenmbers(tfidf_matrixt,coms,centroid,removed_nodes,MIN_SIMILARITY)
        #重新计算质心
        #centroid=getCentroid(tfidf_matrixt,coms)
        #以新的质心继续迭代
        Propagate(AdjacentMatrix,tfidf_matrixt,coms,centroid,left_nodes,removed_nodes,MIN_SIMILARITY)
    else:
        return coms

##################################################################################
#最终的调用接口
##################################################################################
def clusterByCommunityDetection(all_sentences_list,tfidf_matrixt,MIN_SIMILARITY,MAX_SIMILARITY):
    #发现词之间的社区，对词进行聚类
    print "start to detect commnuity"

    #生成图的矩阵
    #按照关键词计算相似性
    #AdjacentMatrix,num_nodes=getAdjacentMatrixForSentences_tfidf(all_sentences_list,word_tfidf_dic)
    #直接按照tfidf向量计算相似性
    AdjacentMatrix=getAdjacentMatrixForSentences_tfidfvector(tfidf_matrixt)

    #基于社区划分的聚类
    graph = getCommunity_BySentences(AdjacentMatrix,tfidf_matrixt,MIN_SIMILARITY,MAX_SIMILARITY)
    
    #返回社区的列表
    community_list=graph.values()

    return community_list


##################################################################################
#去除远离质心的节点
##################################################################################           
def removeCommunityMenmbers(tfidf_matrixt,coms,centroid,removed_nodes,MIN_SIMILARITY):
    to_be_removed_list=[]
    for sentence_id in coms:
        distance=caculatCosine(tfidf_matrixt[sentence_id],centroid)
        if distance<MIN_SIMILARITY:
            to_be_removed_list.append(sentence_id)
            removed_nodes.append(sentence_id)
    for sentence_id in to_be_removed_list:
        coms.remove(sentence_id)
        print "node {} is removed".format(sentence_id)


##################################################################################
#在发现一个社区之后，释放社区边缘中的句子，以作为新社区的成员，并将发现社区
#的关键词从已有数据中删除，对已有数据进行降维处理
##################################################################################
def removeCommunityNodes(tfidf_matrixt,coms,left_nodes,MAX_SIMILARITY):
    #计算新社区的中心
    centroid=coms.getCentroid()
    #社区的成员
    community_menmbers_list=coms.getMenmberSentences().keys()

    #释放社区边缘的句子
    for sentence_id in community_menmbers_list:
        distance=caculatCosine(tfidf_matrixt[sentence_id],centroid)
        if distance>MAX_SIMILARITY:
            left_nodes.remove(sentence_id)#标识该句子已经分配好社区，不能在被其他社区选择
            print "a node is removed from left nodes"
        else:
            print "a node is left for other community"
        
    #获取当前社区的关键词
    keywords_list=getKeyWords(centroid)
    #keywords_list=getTopKKeyword(centroid,3)
    #设置社区的关键词
    coms.setDominateWords(keywords_list)

    #从剩余数据中，将当前成员变量的关键词给删除
    node_to_update=[]
    for sentence_id in left_nodes:
        for word_id in keywords_list:
            if tfidf_matrixt[sentence_id][word_id]>0:
                tfidf_matrixt[sentence_id][word_id]=0
                node_to_update.append(sentence_id)
    #将List转成set返回
    return set(node_to_update)
                
                    
                    
                
            

    


##################################################################################
#获取当前社区的所有邻居节点，并他们与当前社区质心的距离
##################################################################################
def getCommunityNeighbours(AdjacentMatrix,tfidf_matrixt,coms,centroid,left_nodes,removed_nodes):
    #先获取当前社区的所有邻居节点
    nel = getAllneighbours(AdjacentMatrix,coms,left_nodes,removed_nodes)
    nelf= []
    
    #计算邻居节点中与质心距离
    if nel:
        for node_id in nel:
            distance=caculatCosine(tfidf_matrixt[node_id],centroid)
            nelf.append(distance)
    return nel,nelf


#获取一个社区的所有邻居节点
def getAllneighbours(AdjacentMatrix,coms,left_nodes,removed_nodes):
    ne = []
    for node_id in coms:
        for j in range(0,len(AdjacentMatrix)):
            if j in left_nodes and AdjacentMatrix[node_id][j]>0:
                if j not in coms and j not in removed_nodes and j not in ne:
                    ne.append(j)
    return ne





##################################################################################
#采用tf-idf将句子向量化，然后采用社区发现方法来聚类
##################################################################################
def clusterByCommunityDetection_tfidfvector(all_sentences_list,word_tfidf_dic,SIZE_COMMUNITY):
    #准备数据
    data=documentsVectorizer(all_sentences_list,word_tfidf_dic)

    #计算相识度矩阵
    num_sentences = len(all_sentences_list)
    AdjacentMatrix=np.zeros((num_sentences,num_sentences))
    for i in range(0, num_sentences):
        set_a=set(all_sentences_list[i])
        for j in range(i+1, num_sentences):
            set_b=set(all_sentences_list[j])
            inter_set=set_a & set_b
            if len(inter_set)>1:
                sim_i_j=caculatCosine(data[i].tolist(), data[j].tolist())
                AdjacentMatrix[i][j]=sim_i_j
                AdjacentMatrix[j][i]=sim_i_j
                
    #计算各个句子的权重
    sentence_weight_list=getSentenceWeights(all_sentences_list,word_tfidf_dic)
    
    degree_s, neighbours, sums = Degree_Sorting(AdjacentMatrix, num_sentences)
    graph = getCommunity_BySentences(AdjacentMatrix,sentence_weight_list,neighbours,degree_s,num_sentences,SIZE_COMMUNITY)

    
    #去除相似度大的社区
    community_list=graph.values()
    mergeCommunity(community_list,MIN_SIMILARITY_FOR_COMMUNITY_MERGE)
    
    return community_list


