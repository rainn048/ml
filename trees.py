# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from math import log
import operator
import pandas as pd
from sklearn.datasets import load_iris
import treePlotter

##########################################
# 1. 计算香农熵
# [DataFrame]dataSet: 数据集最后一列为分类标签
##########################################
def calcShannonEnt(dataSet):
    """计算信息熵.
    
    Parameters
    ----------
    dataSet : dataframe, 样本数据集，最后一列为分类标签

    """
    numEntries = dataSet.shape[0]
    labelCounts = {}
    # 统计所有分类
    for index, row in dataSet.iterrows():
        label = row[-1]

        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    
    #计算信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt



#############################################
#2. 按照给定特征划分数据集
# dataSet[DataFrame]:样本集
# axis[int]:第i个特征
# value[numberic]: 特征值
#############################################
def splitDataSet(dataSet, axis, value):
    """按照指定列分裂数据.
    
    Parameters
    ----------
    dataSet : dataframe, 样本数据集，最后一列为分类标签
    axis: index，特征的列名
    value: float, 特征值
    
    """
#    print("**********")
#    print(dataSet[axis])
#    print("**********")
#    print(value)

    retDataSet = dataSet[dataSet[axis] == value]
    
    return retDataSet



################################################
#3. 选择最好的数据集划分方式
#[DataFrame] dataSet: 样本集
#
################################################

def chooseBestFeatureToSplit(dataSet):
    """选择最好的划分特征.
    
    Parameters
    ----------
    dataSet : dataframe, 样本数据集，最后一列为分类标签
    
    """
    
    
    shape = dataSet.shape
    
    baseEntropy = calcShannonEnt(dataSet) #基础信息熵
    
    bestInfoGain = 0.0; bestFeature = -1 
    for i,col in dataSet.iloc[:,0:-1].iteritems():
        eachFeat = dataSet.loc[:, i].drop_duplicates()
        newEntropy = 0.0
        for index, value in eachFeat.iteritems():
            subDataSet = splitDataSet(dataSet, i, value) #按照当前特征划分数据集
            prob = subDataSet.shape[0]/float(shape[0])
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature



##########################################################
#4. 多数表决分类
##########################################################

def majorityCnt(classList):
    """多数投票选择分类结果.
    
    Parameters
    ----------
    classList : Series, 分类标签
    
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse = True)
    return sortedClassCount[0][0]

    
##########################################################
#5. 创建决策树
# [DataFrame] dataSet: 样本
# [Series] label：分类标签
##########################################################   
def createTree(dataSet):
    """创建决策树.
    
    Parameters
    ----------
    dataSet : dataframe, 样本数据集，最后一列为分类标签
    
    """
    
    classList = dataSet.iloc[:,-1]
    if classList.drop_duplicates().size == 1:#1. 类别完全相同则停止继续划分
        #print("1. 类别完全相同则停止继续划分: ", classList.iloc[0])
        return classList.iloc[0]

    if dataSet.shape[0] == 1:
        #print("遍历完所有特征,返回出现次数最多的类别.")
        return majorityCnt(classList)#遍历完所有特征时返回出现次数最多的得到列表包含的所有属性值
        
    bestFeat = chooseBestFeatureToSplit(dataSet)
    
    myTree = {bestFeat:{}}
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", bestFeat)
    featValues = dataSet[bestFeat]
    uniqueVals = featValues.drop_duplicates()
    for index,value in uniqueVals.iteritems():
        #print("**v***:",value)
        myTree[bestFeat][value] = createTree(splitDataSet(dataSet, bestFeat, value))
    print("++++++++++++++++++++++++++++++++++++++", bestFeat)
    return myTree     
    
    
##########################################################
#5. 决策树
##########################################################    
def main():
#    raw = pd.read_csv(r'C:\Users\LEE\Desktop\5.csv', sep=',', encoding='gb2312')
#    raw = pd.DataFrame(raw, columns=['nTime', 'THour0','THour1','THour2',
#                                   'THour3', 'THour4', 'THour5', 'THour6','THour7','THour8',
#                                   'THour9', 'THour10', 'THour11', 'THour12','THour13','THour14',
#                                   'THour15', 'THour16', 'THour17', 'THour18','THour19',
#                                   'THour20', 'THour21', 'THour22', 'THour23', 'line'])
    iris = load_iris()

    raw = pd.DataFrame(iris.data, columns=iris.feature_names)
    raw['target'] = iris.target
    
    #raw1 = raw.sample(10)
    #print(raw1.iloc[:,-1])
    myTree = createTree(raw)
    #print(myTree)
    treePlotter.createPlot(myTree)
    

if __name__ =="__main__":
    main()   
    
    
    
    
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
