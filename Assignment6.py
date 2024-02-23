import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans

#1
df = pd.read_csv('C:/Users/josep/Documents/IU/Completed Courses/Python ML/Assignment6/faithful.csv')
df = df[["eruptions","waiting"]].astype(float)

naCheck = df.isna().any()
print(naCheck)

df.plot(x="eruptions", y="waiting", style='o')
plt.show()

X = df.values

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = '1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = '2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = '3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'cyan', label = '4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of eruptions')
plt.xlabel('Eruptions')
plt.ylabel('Waiting')
plt.legend()
plt.show()


#2
dfiris = pd.read_csv('C:/Users/josep/Documents/IU/Completed Courses/Python ML/Assignment6/IRIS.csv')
print(dfiris.head(3))
x = dfiris.iloc[:, :-1].values
y = dfiris.iloc[:, -1].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
pca = PCA(n_components=2)
xtrain = pca.fit_transform(xtrain)
xtest = pca.transform(xtest)
classifier = LogisticRegression(random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
print(accuracy_score(ytest, ypred))
xset, yset = xtrain, ytrain
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.legend()
plt.show()

#3

dfSynth = pd.read_csv('C:/Users/josep/Documents/IU/Completed Courses/Python ML/Assignment6/synthetic_control.csv', header=None)
dfSynth['class'] = ''
dfSynth['class'][:100] = "0"
dfSynth['class'][100:200] = "1"
dfSynth['class'][200:300] = "2"
dfSynth['class'][300:400] = "3"
dfSynth['class'][400:500] = "4"
dfSynth['class'][500:] = "5"

naCheck = dfSynth.isna().any()
print(naCheck)

x = dfSynth.iloc[:, :-1].values
y = dfSynth.iloc[:, -1].values.astype(int)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .3, random_state = 0)

pca = PCA(n_components=10)
sc = StandardScaler()
xtrain = pca.fit_transform(xtrain)
xtest = pca.transform(xtest)
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
xk = pca.transform(x)

#kmeans
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(xk)

plt.scatter(xk[y_kmeans == 0, 0], xk[y_kmeans == 0, 1], s = 50, c = 'red', label = '0')
plt.scatter(xk[y_kmeans == 1, 0], xk[y_kmeans == 1, 1], s = 50, c = 'blue', label = '1')
plt.scatter(xk[y_kmeans == 2, 0], xk[y_kmeans == 2, 1], s = 50, c = 'green', label = '2')
plt.scatter(xk[y_kmeans == 3, 0], xk[y_kmeans == 3, 1], s = 50, c = 'yellow', label = '3')
plt.scatter(xk[y_kmeans == 4, 0], xk[y_kmeans == 4, 1], s = 50, c = 'brown', label = '4')
plt.scatter(xk[y_kmeans == 5, 0], xk[y_kmeans == 5, 1], s = 50, c = 'purple', label = '5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 40, c = 'black', label = 'Centroids')
plt.title('KMeans Visualization')
plt.legend()
plt.show()
print(np.concatenate((y_kmeans.reshape(len(y_kmeans),1), y.reshape(len(y),1)),1))
cm = confusion_matrix(y, y_kmeans)
print(cm)
print(accuracy_score(y, y_kmeans))


#SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))
cm = confusion_matrix(ytest, y_pred)
print(cm)
print(accuracy_score(ytest, y_pred))
xset, yset = xtrain, ytrain
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'green', 'blue', 'yellow','brown','purple'))(i), label = j)
plt.legend()
plt.title('SVM Visualization')
plt.show()

#Naive Bayes
classifier = GaussianNB()
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))
cm = confusion_matrix(ytest, y_pred)
print(cm)
print(accuracy_score(ytest, y_pred))
xset, yset = xtrain, ytrain
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'green', 'blue', 'yellow','brown','purple'))(i), label = j)
plt.legend()
plt.title('Naive Bayes Visualization')
plt.show()

#logistic regression
classifier = LogisticRegression(random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))
cm = confusion_matrix(ytest, ypred)
print(cm)
print(accuracy_score(ytest, ypred))
xset, yset = xtrain, ytrain
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'green', 'blue', 'yellow','brown','purple'))(i), label = j)
plt.legend()
plt.title('Logistic Regression Visualization')
plt.show()