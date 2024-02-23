import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def SSE(given, predicted):
     n=len(given)
     SSE=((np.asarray(given)-np.asarray(predicted))**2)/float(2*n)
     return sum(SSE)

def polyfitting(x, y):
     xcolumn = str(x.columns.values)
     poly = PolynomialFeatures(degree = 5)
     xpoly = poly.fit_transform(x)
     poly_reg = LinearRegression()
     poly_reg.fit(xpoly, y)
     plt.scatter(x, y, color = 'blue')
     plt.plot(x, poly_reg.predict(xpoly), color = 'red')
     title = xcolumn + " vs Power Consumption"
     plt.title(title)
     plt.xlabel(xcolumn)
     plt.ylabel("Power Consumption")
     plt.show()
     y_pred = poly_reg.predict(xpoly)
     print("Predicted Values: ",y_pred)
     SSEvar = SSE(y,y_pred)
     print("SSE:",SSEvar)

def multifitting(x, y):
     xcolumn = str(x.columns.values)
     multi_reg = LinearRegression()
     multi_reg.fit(x, y)
     plt.scatter(x, y, color = 'blue')
     plt.plot(x, multi_reg.predict(x), color = 'red')
     title = xcolumn + " vs Power Consumption"
     plt.title(title)
     plt.xlabel(xcolumn)
     plt.show()
     y_pred = multi_reg.predict(x)
     np.set_printoptions(precision=2)
     print(np.concatenate((np.reshape(y_pred,newshape=(len(y_pred),1)), np.reshape(y,newshape=(len(y),1))),1))     
     #print("Predicted Values: ",y_pred)
     SSEvar = SSE(y,y_pred)
     print("SSE:", SSEvar)

data = pd.read_csv('C:/Users/josep/Documents/IU/InProgressCourses/Python ML/Assignment2/Electricity_Consumption-1.csv')
df = data.drop(["DateTime"],axis=1)
matrix = df.corr()
xlab = list(df.columns)
ylab = list(df.columns)
sns.heatmap(matrix, cmap="RdPu", xticklabels=xlab, yticklabels=ylab)
plt.show()

sns.heatmap(matrix, cmap="RdPu", xticklabels=xlab, yticklabels=ylab, vmin=.4, vmax=1)
plt.show()

x = df.loc[:, ["Temperature"]]
y = df.loc[:, ["Power Consumption"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1/3, random_state = 0)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

y_pred = regressor.predict(xtest)
print("Predicted Values: ",y_pred)
SSEvar = SSE(ytest,y_pred)
print("SSE:",SSEvar)
plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Temperature vs Power Consumption")
plt.xlabel("Temperature")
plt.ylabel("Power Consumption")
plt.show()

x = df.loc[:, ["general diffuse flows"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1/3, random_state = 0)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

y_pred = regressor.predict(xtest)
print("Predicted Values: ",y_pred)
SSEvar = SSE(ytest,y_pred)
print("SSE:",SSEvar)
plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("general diffuse flows vs Power Consumption")
plt.xlabel("general diffuse flows")
plt.ylabel("Power Consumption")
plt.show()

x = data.loc[:, ["Temperature"]]
polyfitting(x,y)

x = df.loc[:, ["general diffuse flows"]]
polyfitting(x,y)


x = data.loc[:, ["Temperature"]]
multifitting(x,y)

x = df.loc[:, ["general diffuse flows"]]
multifitting(x,y)