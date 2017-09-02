# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:43:57 2017

@author: LEE
"""

import numpy as np
import pandas as pd


def loadDataSet():
    postingList=pd.Series([['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']])
    classVec = pd.Series([0,1,0,1,0,1])    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):
    """创建不重复词列表.
        
        Parameters
        ----------
        dataSet : Series, 输入数据源
        
        Return
        ----------
        vocabSet : Series, 不重复词列表
        
    """
        
    vocabSet = set([])  #create empty set
    
    for index, value in dataSet.iteritems():
        vocabSet = vocabSet | set(value) #union of the two sets
    return pd.Series(list(vocabSet))
    

def setOfWords2Vec(vocabList, input):
    """创建不重复词列表.
        
        Parameters
        ----------
        vocabList : Series, 词汇表
        input：  Series，某个词文档
        
        Return
        ----------
        vocabSet : Series, 词向量
        
    """
    
    
    returnVec = pd.Series(np.zeros(vocabList.size, dtype=np.int))#初始化一个全零的Series
    returnVec[vocabList.isin(input)] = 1 #如果input中的单词在vocabList，则returnVec对应位置赋1
    
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)#样本个数
    numWords=len(trainMatrix[0])#文档中单词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)#1文档比例
    pONum = zeros(numWords); plNum = zeros(numWords) #初始化概率
    pODenom = 0.0; plDenom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            plNum += trainMatrix[i] #向 量 相 加
            plDenom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i] )
    plVect = plNum / plDenom #changetolog{) <3—n
    p0Vect = p0Num / p0Denom #change to log() 办对每个元素做除法
    return p0Vect,plVect ,pAbusive    
    
    
    
    
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
s = pd.Series(listOPosts[0])
returnVec =setOfWords2Vec(myVocabList, s)
returnVec