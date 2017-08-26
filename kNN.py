# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:49:00 2017

@author: LEE
"""

import pandas as pd
import matplotlib.pyplot as plt


def test():
    raw = pd.read_csv(r'D:\machineLearn\git\ml\datingTestSet.txt', sep='\t', 
                  encoding='utf8', header=None)
    
    raw[3] = raw[3].map({'largeDoses':3, 'smallDoses':2, 'didntLike':1})
    
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    ax.scatter(raw.iloc[:,1],raw.iloc[:,2])
    plt.show()

    return raw



def autoNorm(df):
    minV = df.min()
    maxV = df.max()
    rangeV = maxV -minV
    df = df.sub(minV)
    df = df.div(rangeV)
    
    return df,rangeV,minV
    
    

def classfiy0(X, dataSet, k):
    diff = dataSet.iloc[:, :-1].sub(X)
    sqDiff = diff.pow(2)
    sqDis = sqDiff.sum(axis=1)
    distance =  sqDis.pow(0.5)
    sortedDis = distance.argsort()
    kIndex = sortedDis.head(k)
    labelCount = dataSet.iloc[:,-1].take(kIndex)
    classCount = labelCount.value_counts()
    return classCount.first_valid_index()
        


def datingTest(df):
    normDf, ranges, minV = autoNorm(df)
    numTest = 200
    errorCount = 0.0
    
    for i in range(numTest):
        classResult = classfiy0(normDf.iloc[i,:-1], normDf.iloc[numTest:], 3)
        realResult = normDf.iloc[i, -1]
        print('the classifier came back with： %d, the real answer is: % d .\n'
              %(classResult, realResult))
        if (classResult != realResult):
            errorCount +=1.0
    print("the total error rate is: %f" % (errorCount / float (numTest) )  )   


if __name__ =="__main__":
    df = test() 
    datingTest(df)
    