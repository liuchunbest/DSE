import sys

reload(sys)
sys.setdefaultencoding('utf8')


class Community:
    def __init__(self, centroid,menmber_sentencesd_dic):
        self.id = 0
        self.centroid = centroid
        self.name = ''
        self.dominate_words=[]
        self.menmber_sentences_dic=menmber_sentencesd_dic
                       

    def setName(self, name):
        self.name = name

    def setID(self, id):
        self.id = id

    def setCentroid(self, centroid_list):
        self.centroid = centroid_list

    def setMembers(self, members):
        self.members = members

    def setDominateWords(self, dominate_words):
        self.dominate_words = dominate_words

    def getCentroid(self):
        return self.centroid

    def getID(self):
        return self.id

    def getName(self):
        return self.name

    def getDominateWords(self):
        return self.dominate_words
    
    def getMenmberSentences(self):
        return self.menmber_sentences_dic



