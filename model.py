import pandas as pd     #package required to read parse csv files to the python script
from sklearn.svm import LinearSVC   #package that imports a SVM model
from sklearn.feature_selection import RFE #package that imports a RFE model

svm = LinearSVC()

dataset = pd.read_csv('cell_line_expression.csv') #reads the dataset containing cell line expression values

target = pd.read_csv('carboplatin.csv')  #reads the values

print("Dimensions of the input dataset\n")
print(dataset.shape)
print("Dimensions of the target values\n")
print(target.shape)

X=dataset.iloc[:,0:27912] #creates a matrix containing the input values

print(X.shape) #dimension of the matrix

Y=target.iloc[:,1:2] #creates a vector containing the target values

print(Y.shape)

print(Y)

rfe = RFE(svm, 27900) #specifies which feature ranking metod is to be used and how many features we want the model to be reduced to.

rfe = rfe.fit(X, Y) #fits the model


print(rfe.support_)
print(rfe.ranking_)
