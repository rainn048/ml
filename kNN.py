# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:49:00 2017

@author: LEE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

def test2():
    raw = pd.read_csv(r'D:\machineLearn\git\ml\datingTestSet.txt', sep='\t', 
                  encoding='utf8', header=None)
    
    raw[3] = raw[3].map({'largeDoses':3, 'smallDoses':2, 'didntLike':1})
    
    datingTest(raw)
    
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    ax.scatter(raw.iloc[:,1],raw.iloc[:,2])
    plt.show()

    return raw


###############################################################################
# 0. 构造函数
###############################################################################
def autoNorm(df):
    """归一化
        
        Parameters
        ----------
        df : Dataframe, 归一化df, df=>[0,1]
        
        Returns
        -------
        df : Dataframe,  归一化后的df
        rangeV: Series， 每列的区间
        minV:   Series， 每列的最小值   
            
    """
    
    minV = df.min()
    maxV = df.max()
    rangeV = maxV -minV
    df = df.sub(minV)
    df = df.div(rangeV)
    
    return df,rangeV,minV
    
    

def classify0(X, dataSet, k):
    """k-近邻算法
        
        Parameters
        ----------
        X : Series, 预测的输入样本
        dataSet ： DataFrame, 输入的训练样本集，最后一行为分类标签
        k: int, 选择最近邻居的数据
        
        Returns
        -------
        label : 预测样本的分类标签
            
    """
    
    
    diff = dataSet.iloc[:, :-1].sub(X)
    sqDiff = diff.pow(2)
    sqDis = sqDiff.sum(axis=1)
    distance =  sqDis.pow(0.5)
    sortedDis = distance.argsort()
    kIndex = sortedDis.head(k)
    labelCount = dataSet.iloc[:,-1].take(kIndex)
    classCount = labelCount.value_counts()
    return classCount.first_valid_index()
        

        
def img2vector(filename):
    returnVect = np.zeros(1024)
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[32 * i + j] = int(lineStr[j])
    return returnVect      

def datingTest(df):
    normDf, ranges, minV = autoNorm(df)
    numTest = 200
    errorCount = 0.0
    
    for i in range(numTest):
        classResult = classify0(normDf.iloc[i,:-1], normDf.iloc[numTest:], 3)
        realResult = normDf.iloc[i, -1]
        print('the classifier came back with： %d, the real answer is: % d .\n'
              %(classResult, realResult))
        if (classResult != realResult):
            errorCount +=1.0
    print("the total error rate is: %f" % (errorCount / float (numTest) )  )   


def train():
    hwLabels = []
    trainingFileList = listdir(r'D:\machineLearn\git\ml\digits\trainingDigits') #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros( (m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        
        
        
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r'D:\machineLearn\git\ml\digits\trainingDigits\%s' % fileNameStr)
    
    dataSet = pd.DataFrame(trainingMat)
    dataSet['label'] = hwLabels
    return dataSet


def test(dataSet):
    testFileList = listdir(r'D:\machineLearn\git\ml\digits\testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'D:\machineLearn\git\ml\digits\testDigits\%s' % fileNameStr)
        test = pd.Series(vectorUnderTest)

        classifierResult = classify0(test, dataSet, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))

def handwritingClassTest():
    
    hwLabels = []
    trainingFileList = listdir(r'D:\machineLearn\git\ml\digits\trainingDigits') #load the training set
    m = len(trainingFileList)

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        trainingMat = np.zeros( (m,1024))
        
        
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r'D:\machineLearn\git\ml\digits\trainingDigits\%s' % fileNameStr)
    
    dataSet = pd.DataFrame(trainingMat)
    dataSet['label'] = hwLabels
    
    testFileList = listdir(r'D:\machineLearn\git\ml\digits\testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    mTest=1
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'D:\machineLearn\git\ml\digits\testDigits\%s' % fileNameStr)
        test = pd.Series(vectorUnderTest)
        print(test)
        classifierResult = classify0(test, dataSet, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


#if __name__ =="__main__":
train=train()
test(train) 
    
    