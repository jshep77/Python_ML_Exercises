# Assignment 1
# Joseph Shepherd
# Machine Learning With Python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("C:/Users/josep/Documents/IU/InProgressCourses/Python ML/Assignment1/Forecast_Data_Set.csv")
print("There are", len(list(data)), "lines of data in the file.")
print("The First Ten Rows:\n", data.head(10))
print("The Last Ten Rows:\n", data.tail(10))

# Dropping the LDAPS_PPT4 column from the data set
data = data.drop(["LDAPS_PPT4"], axis=1)

# renaming the Date column to specifically Year_Month_Day and then double-checking the headers
data = data.rename(columns={"Date": "Year_Month_Day"})
print(data.head(1))

# creating and calculating the Next_Taverage column in the data set
data['Next_Taverage'] = data[['Next_Tmin', 'Next_Tmax']].mean(axis=1)

# using the min-max method to normalize the Solar radiation column between 0 and 1 and checking the outcome
data["Solar radiation"] = (data["Solar radiation"] - data["Solar radiation"].min()) / (
            data["Solar radiation"].max() - data["Solar radiation"].min())
print(data["Solar radiation"].min(), data["Solar radiation"].mean(), data["Solar radiation"].max())

#checking if any data is missing in the data set by column
naCheck = data.isna().any()
print(naCheck)

#handling missing data by dropping it entirely if the missing data is in the Year_Month_Day column,
# as we don't know when it should reside in the data, and it is the only non-float/integer field
print(data.iloc[-1])
data = data[data["Year_Month_Day"].notna()]
print("\n",data.iloc[-1])

#handling the remaining missing data by filling it in with zeros via the fillna() method
print(data["LDAPS_WS"][1030:1040])
data = data.fillna(0)
print(data["LDAPS_WS"][1030:1040])

#create a correlation matrix with the predictor variables to be used in the resulting heatmap to compare correlation values among multiple columns
df = data.drop(["Year_Month_Day","Next_Taverage","Next_Tmin","Next_Tmax"], axis=1)
matrix = df.corr()
xlab = list(data.drop(["Year_Month_Day","Next_Taverage","Next_Tmin","Next_Tmax"], axis=1).columns)
ylab = list(data.drop(["Year_Month_Day","Next_Taverage","Next_Tmin","Next_Tmax"], axis=1).columns)
sns.heatmap(matrix, cmap="Greys", xticklabels=xlab, yticklabels=ylab)
plt.show()

#display a reduced heat map with just the strongly correlated columns
sns.heatmap(matrix, cmap="Greys", xticklabels=xlab, yticklabels=ylab, vmin=.6, vmax=1)
plt.show()

#plot scatter charts of each variable with more than .6 correlation.
plt.scatter(data["Present_Tmin"],data["Present_Tmax"])
plt.xlabel("Present_Tmin")
plt.ylabel("Present_Tmax")
plt.show()
plt.scatter(data["LDAPS_Tmax_lapse"],data["LDAPS_Tmin_lapse"])
plt.xlabel("LDAPS_Tmax_lapse")
plt.ylabel("LDAPS_Tmin_lapse")
plt.show()
plt.scatter(data["LDAPS_CC1"],data["LDAPS_CC2"])
plt.xlabel("LDAPS_CC1")
plt.ylabel("LDAPS_CC2")
plt.show()
plt.scatter(data["LDAPS_CC2"],data["LDAPS_CC3"])
plt.xlabel("LDAPS_CC2")
plt.ylabel("LDAPS_CC3")
plt.show()
plt.scatter(data["LDAPS_CC3"],data["LDAPS_CC4"])
plt.xlabel("LDAPS_CC3")
plt.ylabel("LDAPS_CC4")
plt.show()
plt.scatter(data["LDAPS_RHmin"],data["LDAPS_CC1"])
plt.xlabel("LDAPS_RHmin")
plt.ylabel("LDAPS_CC1")
plt.show()
plt.scatter(data["LDAPS_RHmin"],data["LDAPS_CC2"])
plt.xlabel("LDAPS_RHmin")
plt.ylabel("LDAPS_CC2")
plt.show()
plt.scatter(data["LDAPS_RHmin"],data["LDAPS_CC3"])
plt.xlabel("LDAPS_RHmin")
plt.ylabel("LDAPS_CC3")
plt.show()
plt.scatter(data["DEM"],data["Slope"])
plt.xlabel("DEM")
plt.ylabel("Slope")
plt.show()

#Dropping the correlated variables
dfClean = df.drop(["LDAPS_CC1","LDAPS_CC2","LDAPS_CC3"], axis=1)