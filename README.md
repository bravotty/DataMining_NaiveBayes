# DataMining_NaiveBayes

[![auc][aucsvg]][auc] [![License][licensesvg]][license]

[aucsvg]: https://img.shields.io/badge/tyty-NaiveBayes-orange.svg
[auc]: https://github.com/bravotty/DataMining_NaiveBayes

[licensesvg]: https://img.shields.io/badge/License-MIT-blue.svg
[license]: https://github.com/bravotty/DataMining_NaiveBayes/blob/master/LICENSE


```
A python implementation of NaiveBayes
Env     : Python 2.6
```

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
-- comments in code file

class NaiveBayes(object):
	def __init__(self, train=None, trainLabel=None):
		self.train = pd.DataFrame(train)
		self.labels = list(set(trainLabel))
		col = len(self.train.columns)
		self.train.insert(col, col, trainLabel)
		self.classificationGroup = self.train.groupby(self.train.iloc[:, -1])
		self.mean = self.classificationGroup.mean()
		self.variance  = self.classificationGroup.var()


NB = NaiveBayes(trainData, trainDataLabel)

```




## License

[The MIT License](https://github.com/bravotty/DataMining_NaiveBayes/blob/master/LICENSE)