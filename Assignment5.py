import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

def SSE(given, predicted):
    n=len(given)
    SSE=((np.asarray(given)-np.asarray(predicted))**2)/float(2*n)
    return sum(SSE)

def MSE(given, predicted):
    n=len(given)
    MSE=((np.asarray(given)-np.asarray(predicted))**2)/n
    return sum(MSE)


df = pd.read_csv('C:/Users/josep/Documents/IU/InProgressCourses/Python ML/Assignment5/Sales_Properties.csv')

df['Postal_Code'] = pd.factorize(df['Postal_Code'])[0] + 1
df["Property_Type"].replace(["house","unit"], [1,0], inplace=True)
split = df['Date_Property_Sold'].str.split('/', expand=True)
df['Date_Property_Sold_Month'] = split[0].astype('int64')
split_year = split[2].str.split(' ',expand=True)
df['Date_Property_Sold_Year'] = split_year[0].astype('int64')
df = df.drop(["Date_Property_Sold"],axis=1)
df['Date_Property_Sold_Year'] = pd.factorize(df['Date_Property_Sold_Year'])[0] + 1

df = df[['Date_Property_Sold_Month','Date_Property_Sold_Year','Property_Type','Number_Bedrooms','Postal_Code','Property_Price']]

matrix = df.corr()
xlab = list(df.columns)
ylab = list(df.columns)
sns.heatmap(matrix, cmap="RdPu", xticklabels=xlab, yticklabels=ylab, annot=True)
plt.show()

naCheck = df.isna().any()
print(naCheck)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .5, random_state = 0)

#Linear Regression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)
predprice = regressor.predict(xtest)
plt.scatter(xtrain[:,0], ytrain, color = 'red')
plt.scatter(xtrain[:,0], regressor.predict(xtrain), color = 'blue')
plt.legend(["Training Data","Predicited Prices"])
plt.show()
print("Predicted Prices: ", predprice)
print("Sum of Squares Error:",SSE(ytest,predprice))
print("Error Mean Sum of Squares: ", MSE(ytest,predprice))

#Decision Tree
dtree = tree.DecisionTreeRegressor(min_samples_split=50)
dtree.fit(xtrain, ytrain)
predprice2 = dtree.predict(xtest)
plt.scatter(xtrain[:,0], ytrain, color = 'red')
plt.scatter(xtrain[:,0], dtree.predict(xtrain), color = 'blue')
plt.legend(["Training Data","Predicited Prices"])
plt.show()
print("Predicted Prices: ", predprice2)
print("Sum of Squares Error:",SSE(ytest,predprice2))
print("Error Mean Sum of Squares: ", MSE(ytest,predprice2))

#Random Forest
rforest = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, random_state=0)
rforest.fit(xtrain, ytrain)
treePrice = rforest.predict(xtest)
plt.scatter(xtrain[:,0], ytrain, color = 'red')
plt.scatter(xtrain[:,0], rforest.predict(xtrain), color = 'blue')
plt.legend(["Training Data","Predicited Prices"])
plt.show()
print("Predicted Prices: ", treePrice)
print("Sum of Squares Error:",SSE(ytest,treePrice))
print("Error Mean Sum of Squares: ", MSE(ytest,treePrice))


#SHAP work:
explainer1 = shap.Explainer(regressor.predict, xtest)
shap_values = explainer1(xtest)
shap.summary_plot(shap_values, feature_names=xlab)

explainer2 = shap.Explainer(dtree.predict, xtest)
shap_values2 = explainer2(xtest)
shap.summary_plot(shap_values2, feature_names=xlab)

explainer3 = shap.Explainer(rforest.predict, xtest)
shap_values3 = explainer3(xtest)
shap.summary_plot(shap_values3, feature_names=x.columns.tolist())