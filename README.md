# DataMining_NaiveBayes

[![auc][aucsvg]][auc] [![License][licensesvg]][license]

[aucsvg]: https://img.shields.io/badge/tyty-NaiveBayes-orange.svg
[auc]: https://github.com/bravotty/DataMining_NaiveBayes

[licensesvg]: https://img.shields.io/badge/License-MIT-blue.svg
[license]: https://github.com/bravotty/DataMining_NaiveBayes/blob/master/LICENSE


A python implementation of NaiveBayes
Env       : Python 2.6

## Usage     : 

```lisp
	python 2.6
	python naiveBayes.py

```


## Defination :
```lisp
-- Use pandas DataFrame datatype to handle the NaiveBayes Model
-- mean     -> dataFrame.mean()
-- variance -> dataFrame.variance()
class NaiveBayes:
	def __init__(self, train=None, trainLabel=None, mean=None, variance=None, classificationGroup=None):
		self.train = train 				#train data
		self.trainLabel = trainLabel 			#train data label
		self.mean = mean 				#train data mean table (PS:GROUPED)
		self.variance = variance 			#train data variance table  (PS:GROUPED)
		self.classificationGroup = classificationGroup 	#GROUP operation - pd.DataFrame
```