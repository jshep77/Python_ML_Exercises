import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

path = os.getcwd() + '\Assignment3\Raisin_Dataset.csv'
dataset = pd.read_csv(path)

naCheck = dataset.isna().any()
print(naCheck)


#replacing categorical data so that Besni is now class 0 and Kecimen is class 1
dataset["Class"].replace(["Besni","Kecimen"], [0,1], inplace=True)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#problem A 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2, random_state = 0)

#Problem B
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(xtrain, ytrain)

ypred = classifier.predict(xtest)
print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))

conmatrix = confusion_matrix(ytest, ypred)
print("Confusion matrix:", conmatrix)
print("Accuracy Score", accuracy_score(ytest, ypred))

xset, yset = sc.inverse_transform(xtrain), ytrain
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Raisin Class Prediction')
plt.legend()
plt.show()


#problem C
accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 15)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 30)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 300)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


#problem D
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .5, random_state = 0)

print(xtrain, "\n", xtest)

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(xtrain, ytrain)

ypred = classifier.predict(xtest)
print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))

conmatrix = confusion_matrix(ytest, ypred)
print("Confusion matrix:", conmatrix)
print("Accuracy Score", accuracy_score(ytest, ypred))

xset, yset = sc.inverse_transform(xtrain), ytrain
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Raisin Class Prediction')
plt.legend()
plt.show()