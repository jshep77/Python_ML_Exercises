import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import seaborn as sns

trainingdata = pd.read_csv('C:/Users/josep/Documents/IU/InProgressCourses/Python ML/Mini Project/model.csv')
testingdata = pd.read_csv('C:/Users/josep/Documents/IU/InProgressCourses/Python ML/Mini Project/val.csv')

naCheck = trainingdata.isna().any()
print(naCheck)
naCheck = testingdata.isna().any()
print(naCheck)

matrix = trainingdata.corr()
xlab = list(trainingdata.columns)
ylab = list(trainingdata.columns)
sns.heatmap(matrix, cmap="RdPu", xticklabels=xlab, yticklabels=ylab, vmin=.5)
plt.show()

matrix = testingdata.corr()
xlab = list(testingdata.columns)
ylab = list(testingdata.columns)
sns.heatmap(matrix, cmap="RdPu", xticklabels=xlab, yticklabels=ylab, vmin=.5)
plt.show()

X = trainingdata.values
xtrain = trainingdata.iloc[:, :-1].values
ytrain = trainingdata.iloc[:, -1].values
xtest = testingdata.iloc[:, :-1].values
ytest = testingdata.iloc[:, -1].values

#Logistic Regression
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
pca = PCA(n_components=15)
xtrain = pca.fit_transform(xtrain)
xtest = pca.transform(xtest)
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
print(accuracy_score(ytest, ypred))
xset, yset = xtest, ypred
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression Visualization')
plt.legend()
plt.show()

#Kmeans
pca = PCA(n_components=15)
X = pca.fit_transform(X)
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = '0')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = '1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('KMeans Visualization')
plt.legend()
plt.show()
cm = confusion_matrix(ytrain, y_kmeans)
print(cm)
print(accuracy_score(ytrain, y_kmeans))

#retrying the same models but with reduced data for the model training.
X1 = trainingdata.values
X1 = np.delete(X1, range(4000,90001), 0)
xtrain1 = trainingdata.iloc[:, :-1].values
xtrain1 = np.delete(xtrain1, range(4000,90001), 0)
ytrain1 = trainingdata.iloc[:, -1].values
ytrain1 = np.delete(ytrain1, range(4000,90001), 0)
xtest = testingdata.iloc[:, :-1].values
ytest = testingdata.iloc[:, -1].values

#Logistic Regression
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain1)
xtest = sc.transform(xtest)
pca = PCA(n_components=15)
xtrain1 = pca.fit_transform(xtrain1)
xtest = pca.transform(xtest)
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain1, ytrain1)
ypred = classifier.predict(xtest)
df = pd.DataFrame(ypred)
df.to_csv("Mini Project/results1.csv", header=False,index=False)
cm = confusion_matrix(ytest, ypred)
print(cm)
print(accuracy_score(ytest, ypred))
xset, yset = xtest, ypred
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression Reduced Data Visualization')
plt.legend()
plt.show()

#Kmeans
pca = PCA(n_components=15)
X1 = pca.fit_transform(X1)
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
ykmeans = kmeans.fit_predict(X1)
df = pd.DataFrame(ykmeans)
df.to_csv("Mini Project/results2.csv", header=False,index=False)
plt.scatter(X1[ykmeans == 0, 0], X1[ykmeans == 0, 1], s = 20, c = 'red', label = '0')
plt.scatter(X1[ykmeans == 1, 0], X1[ykmeans == 1, 1], s = 20, c = 'blue', label = '1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('KMeans Reduced Data Visualization')
plt.legend()
plt.show()
cm = confusion_matrix(ytrain1, ykmeans)
print(cm)
print(accuracy_score(ytrain1, ykmeans))