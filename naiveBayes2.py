# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-6
# Env    : python2.6

from __future__ import division
import pandas as pd
import tools as tl
import math

class NaiveBayes:
	def __init__(self, train=None, trainLabel=None, mean=None, variance=None, classificationGroup=None):
		self.train = train                                #train data
		self.trainLabel = trainLabel 					  #train data label
		self.mean = mean 								  #train data mean table (PS:GROUPED)
		self.variance = variance  								  #train data variance table  (PS:GROUPED)
		self.classificationGroup = classificationGroup    #GROUP operation - pd.DataFrame

	def fitTransform(self, trainX, trainLabelX):
		#convert list into pd.dataframe and reserve the labels
		self.train = pd.DataFrame(trainX)
		self.labels = list(set(trainLabelX))
		print self.labels
		#cal the column number of trainX, fruit.txt is 4
		#insert the label to 4th col
		col = len(self.train.columns)
		self.train.insert(col, col, trainLabelX)
		#According the label col ----> group by 4 classes
		#In order to cal the mean and variance of the Data
		self.classificationGroup = self.train.groupby(self.train.iloc[:, -1])
		#data mean table and variance table
		#         1     2     3     4
		# class1  0.22  0.34  0.45  0.13
		# class2  ...
		# class3  ...
		# class4  ...
		self.mean = self.classificationGroup.mean()
		self.variance  = self.classificationGroup.var()
	
	#normal distribution
	def normalDistributionCalculateFunction(self, val, mean, variance):
		coff = 1 / (math.sqrt(2 * math.pi * variance))
		exp  = math.exp(- pow(val - mean, 2) / (2 * variance))
		res  = coff * exp
		return res

	def classification(self, trainE):
    	#initial with eg nums of labels
		groupNum = self.classificationGroup.count()
		groupNumLabel = groupNum.iloc[:, -1].tolist();
		print groupNumLabel
		#cal the P(Y) = [..., ... , ..., ...]
		groupProbility = [n / sum(groupNumLabel) for n in groupNumLabel]
		print groupProbility
		for i in range(len(trainE)):
			P = []
			for j in range(len(self.labels)):
				P.append(self.normalDistributionCalculateFunction(trainE[i], self.mean.iloc[j, i], self.variance.iloc[j, i]))
			PX = [groupProbility[a] * P[a] for a in range(len(P))]
		maxIndex = PX.index(max(PX))
		return self.labels[maxIndex]

	#
	def prediction(self, testY):
		predictionLabel = []
		for i in testY:
			predictionLabel.append(self.classification(i))
		return predictionLabel

	#accuracy function
	def accuracy(self, predictionLabel, testLabel):
		cnt = 0
		for i in range(len(testLabel)):
			if predictionLabel[i] == testLabel[i]:
				cnt += 1	
		acc = cnt / len(testLabel)
		return acc

	#recall function
	def recall(self, predictionLabel, testLabel, dataSetlength=59):
		cnt = 0
		for i in range(len(testLabel)):
			if predictionLabel[i] == testLabel[i]:
				cnt += 1
		rec = cnt / dataSetlength
		return rec

	#f-value function
	def Fvalue(self, predictionLabel, testLabel):
		acc = accuracy(predictionLabel, testLabel)
		rec = recall(predictionLabel, testLabel)
		# if the  denominator == 0: ERR
		if (acc + rec) == 0:
			print 'Bad NaiveBayes prediction!'
			return 0
		F   = (acc * rec * 2) / (acc + rec)
		return F


train, trainLabel, test, testLabel = tl.createDataSet()

ctl = NaiveBayes()
ctl.fitTransform(train,trainLabel)
predictionLabel = ctl.prediction(test)
print testLabel
res2 = ctl.accuracy(predictionLabel, testLabel)
print res2


