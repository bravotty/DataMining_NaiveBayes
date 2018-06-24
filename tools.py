# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-6
# Env    : python2.6

from __future__ import division
import pandas as pd
import numpy as np

def createDataSet(splitSize=0.2):
    fruit = pd.read_table('./fruit.txt')
    #convert pd.DataFrame -> ndarray -> list 
    fruit.head()
    #print fruit.shape
    labelsDict = {}
    labels = ['mass', 'width', 'height', 'color_score', 'fruit_label']
    #choose the labels fruits data
    trainData = fruit[labels]
    numpyTrainData = np.array(trainData)
    # dataSet = numpy_train_data.tolist()
    #list - dataSet
    recordNums = numpyTrainData.shape[0]
    trainDataIndex = range(recordNums)
    #train_data_index = [1, ..., 59]
    testDataIndex = []
    testNumber = int(recordNums * splitSize)
    for i in range(testNumber):
    	#choose test_number test e.g.s
    	randomNum = int(np.random.uniform(0, len(trainDataIndex)))
    	testDataIndex.append(trainDataIndex[randomNum])
    	del trainDataIndex[randomNum]
    trainSet = numpyTrainData[trainDataIndex]
    testSet  = numpyTrainData[testDataIndex]
    trainSet = trainSet.tolist()
    testSet  = testSet.tolist()

    trainLabel = [a[-1]  for a in trainSet]
    trainSet   = [a[:-1] for a in trainSet]
    testlabel  = [a[-1]  for a in testSet]
    testSet    = [a[:-1] for a in testSet]

    # for i in range(len(trainSet[0])):
    #     temp = [a[i] for a in trainSet]
    #     maxNumber = max(temp)
    #     minNumber = min(temp)
    #     #standardize the dataSet
    #     for j in range(len(trainSet)):
    #         denominator = maxNumber - minNumber
    #         trainSet[j][i] = (trainSet[j][i] - minNumber) / denominator

    # for i in range(len(testSet[0])):
    #     temp = [a[i] for a in testSet]
    #     maxNumber = max(temp)
    #     minNumber = min(temp)
    #     #standardize the dataSet
    #     for j in range(len(testSet)):
    #         denominator = maxNumber - minNumber
    #         testSet[j][i] = (testSet[j][i] - minNumber) / denominator
    #print trainSet
    #print testSet
    # print testlabel
    return trainSet, trainLabel, testSet, testlabel

trainSet, trainLabel, testSet, testlabel  = createDataSet()

#accuracy function
def accuracy(predictionLabel, testLabel):
    cnt = 0
    for i in range(len(testLabel)):
        if predictionLabel[i] == testLabel[i]:
            cnt += 1    
    acc = cnt / len(testLabel)
    return acc

#recall function
def recall(predictionLabel, testLabel, dataSetlength=59):
    cnt = 0
    for i in range(len(testLabel)):
        if predictionLabel[i] == testLabel[i]:
            cnt += 1
    rec = cnt / dataSetlength
    return rec

#f-value function
def Fvalue(predictionLabel, testLabel):
    acc = accuracy(predictionLabel, testLabel)
    rec = recall(predictionLabel, testLabel)
    # if the  denominator == 0: ERR
    if (acc + rec) == 0:
        print 'Bad NaiveBayes prediction!'
        return 0
    F   = (acc * rec * 2) / (acc + rec)
    return F
    



