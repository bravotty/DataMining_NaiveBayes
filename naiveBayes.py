# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-6
# Env    : python2.6

from __future__ import division
import pandas as pd
import tools as tl
import math

class NaiveBayes(object):
	def __init__(self, train=None, trainLabel=None):
		#convert list into pd.dataframe and reserve the labels
		self.train = pd.DataFrame(train)
		self.labels = list(set(trainLabel))
		# print self.labels
		#cal the column number of trainX, fruit.txt is 4
		#insert the label to 4th col
		col = len(self.train.columns)
		self.train.insert(col, col, trainLabel)
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
	
	#normal distribution calculate function  - > input mean, variance, value
	def normalDistributionCalculateFunction(self, val, mean, variance):
		#cal the coff
		coff = 1 / (math.sqrt(2 * math.pi * variance))
		#cal the exponent
		exp  = math.exp(- pow(val - mean, 2) / (2 * variance))
		#return the result
		res  = coff * exp
		return res

	def classification(self, trainE):
    	#initial with eg nums of labels
		groupNum = self.classificationGroup.count()
		groupNumLabel = groupNum.iloc[:, -1].tolist();
		#cal the P(Y) = [..., ... , ..., ...] 
		groupProbility = [n / sum(groupNumLabel) for n in groupNumLabel]
		# print trainE --> result
		# trainE = [150, 7.1, 7.9, 0.75] list
		# for each trainE[i] cal the NDCF(trainE[i], mean, variance)
		for i in range(len(trainE)):
			P = []
			for j in range(len(self.labels)):
				P.append(self.normalDistributionCalculateFunction(trainE[i], self.mean.iloc[j, i], self.variance.iloc[j, i]))
			#update the groupProbility - index the PX_Y
			groupProbility = [groupProbility[a] * P[a] for a in range(len(P))]
		#final groupPro = PX_Y
		#find the max Probility in PX_Y
		maxProb = groupProbility.index(max(groupProbility))
		return self.labels[maxProb]

	#prediction function -- input testdata-output prediction label
	def prediction(self, testY):
		predictionLabel = []
		#classify each test sample
		for testSample in testY:
			predictionLabel.append(self.classification(testSample))
		return predictionLabel

def NaiveBayesModelMain():
	#from tools to create the train,trainlabel,test,testlabel
	train, trainLabel, test, testLabel = tl.createDataSet()
	#declare the naivebayed model and fit the params of the model with trainSet and trainLabel
	NaiveBayesModel = NaiveBayes(train, trainLabel)
	#test the model with the testData
	predictionLabel = NaiveBayesModel.prediction(test)
	#calculate the acc, rec and F between predict result and testLabel from tools
	acc = tl.accuracy(predictionLabel, testLabel)
	rec = tl.recall(predictionLabel, testLabel)
	F   = tl.Fvalue(predictionLabel, testLabel)
	#print the acc, rec and F
	print 'NaiveBayesModel Accuracy : ' + str(acc)
	print 'NaiveBayesModel Recall   : ' + str(rec)
	print 'NaiveBayesModel F-value  : ' + str(F)

if __name__ == '__main__':
	#main func start
	NaiveBayesModelMain()

