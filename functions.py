import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
     print(np.concatenate((np.reshape(y_pred,newshape=(len(y_pred),1)), np.reshape(y,newshape=(len(y),1))),1))     #print("Predicted Values: ",y_pred)
     SSEvar = SSE(y,y_pred)
     print("SSE:", SSEvar)