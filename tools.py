# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-6
# Env    : python2.6

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
    # print testSet
    # print testlabel
    return trainSet, trainLabel, testSet, testlabel

# trainSet, trainLabel, testSet, testlabel  = createDataSet()


    



