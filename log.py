# coding: utf-8
import sys
import time
reload(sys)  
sys.setdefaultencoding('utf8')

import codecs

from parameters import *


log = codecs.open(path+"log\\log_"+time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))+".txt", "a", encoding='utf-8')

def writelog(text):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]")
    log.write(text)
    log.write("\n")

def writeloglist(vector):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]")
    size=len(vector)
    if size==0:
        return
    log.write("[ ")
    for i in range(0, size):
        log.write(str(vector[i])+"\t")
    log.write("]")
    log.write("\n")


    #输出结果，将聚簇的相关信息写入到文件
def writelogcluster(cluster):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]")
    menmbers=cluster.getMembers()
    log.write('************输出聚簇信息如下***********'+"\n")
    log.write('menmbers of cluster: '+str(cluster.getID())+':'+'\t')
    for item in menmbers.keys():
       log.write(str(item)+'\t')
    log.write('\n')
    #log.write('聚簇的各个成员向量如下'+"\n")
    #for (key, mem) in menmbers.items():
    #    log.write("成员"+str(key)+": ")
     #   for item in mem:
     #      log.write(str(item)+'\t')
     #   log.write('\n')
    #log.write('\n')
    centroid=cluster.getCentroid()#输出质心
    log.write('centroid of cluster: '+str(cluster.getID())+':'+'\t')
    size_centroid=len(centroid)
    for item in centroid:
        log.write(str(item)+'\t')
    log.write('\n')
    dominate_words=cluster.getDominateWords()#输出dominate words
    log.write('dominate words of cluster: '+str(cluster.getID())+':'+'\t')
    for item in dominate_words:
        log.write(str(item)+'\t')
    log.write('\n')

def writelogfeature(feature):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]")
    log.write(str(feature.product_id) + " " + str(feature.id) + " " + str(feature.text) + "\n")

def writelogname(cluster_dic):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]")
    for (id, cluster) in cluster_dic.items():
        log.write("****第"+str(id)+"个聚簇的信息如下******\n")
        log.write('cluster'+str(cluster.getID())+' name:'+'\t'+cluster.getName()+'\n')
        log.write("cluster"+str(cluster.getID())+' dominate words:'+'\t[')
        words=cluster.getDominateWords()
        for item in words:
            log.write(item+"\t")
        log.write("]")
        log.write("\n")


def writelogProductFeatureMap_100(product_feature_dic, cluster_dic):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]\n")
    log.write("******************前100个产品每个产品包含的特征如下**********************\n")
    k=0
    for (product_id, feature_list) in product_feature_dic.items():
        if k>=100:
            break
        
        log.write("**************The "+str(product_id)+" product************\n")
        log.write(str(product_id)+":"+"[")
        for feature_id in feature_list:
            log.write(str(feature_id)+",")
        log.write("]\n")
        for feature_id in feature_list:
            feature_name=cluster_dic[feature_id].getName()
            log.write(str(feature_id)+": ")
            log.write(feature_name)
            log.write("\n")
    #在输出另外一种形式的
    writelogProductFeatureMap_100_2(product_feature_dic, cluster_dic)
            
            
def writelogProductFeatureMap_100_2(product_feature_dic, cluster_dic):
    log.write("["+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"]\n")
    log.write("*********前100个产品所包含的特征如下******\n")
    k=0
    total_feature_list=[]
    for (product_id, feature_list) in product_feature_dic.items():
        if k>=100:
            break

        for feature_id in feature_list:
            if feature_id not in total_feature_list:
                total_feature_list.append(feature_id)
                
    for feature_id in total_feature_list:
        feature_name=cluster_dic[feature_id].getName()
        log.write(str(feature_id)+"：")
        log.write(feature_name)
        log.write("\n")
          
    
    
    


def closelog():
    log.close()