import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


basepath = os.getcwd()
pospath = basepath + '\Assignment4\TrainingDataPositive.txt'
negpath = basepath + '\Assignment4\TrainingDataNegative.txt'
testpath = basepath + '\Assignment4\\testSet.txt'
propospath = basepath + '\Assignment4\TrainingDataPositive1.txt'
pronegpath = basepath + '\Assignment4\TrainingDataNegative1.txt'
protestpath = basepath + '\Assignment4\\testSet1.txt'

def processfile(file,processfile,review):
    file=open(file)
    writeFile=open(processfile,"w")
    badChar="[,!.?#@=\n]"
    for line in file:
        line=line.lower().replace("\t"," ")
        line=re.sub(badChar,"",line)
        arr=line.split(" ")
        words = review + arr
        toWrite= " ".join(word for word in words[0:len(words)]) 
        writeFile.write(toWrite)
        writeFile.write("\n")
    file.close()
    writeFile.close()

def processtestfile(file,processfile):
    i=0
    file=open(file)
    writeFile=open(processfile,"w")
    badChar="[,!.?#@=\n]"
    for line in file:
        line=line.lower().replace("\t"," ")
        line=re.sub(badChar,"",line)
        arr=line.split(" ")
        if (i<2989):
            review = ["positive,"]
        else:
            review = ["negative,"]
        words = review + arr
        toWrite= " ".join(word for word in words[0:len(words)])
        writeFile.write(toWrite)
        writeFile.write("\n")
        i = i + 1
    file.close()
    writeFile.close()

def getDataAndLabel(processedfile):
    file = open(processedfile)
    label=[]
    data=[]
    for line in file:
        arr=line.replace("\n","").split(",")
        label.append(arr[0])
        data.append(arr[1].replace("\n",""))
    return data,label

processfile(pospath,propospath,["positive,"])
processfile(negpath,pronegpath,["negative,"])
processtestfile(testpath,protestpath)

traindata1,trainlabel1 = getDataAndLabel(propospath)
traindata2,trainlabel2 = getDataAndLabel(pronegpath)
testdata, testlabel = getDataAndLabel(protestpath)
traindata = traindata1 + traindata2
trainlabel = trainlabel1 + trainlabel2
xlab = ["Training Label Negative","Training Label Positive"]
ylab = ["Test Label Negative","Test Label Positive"]

#formatting the data to be used in the models
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train_counts = count_vect.fit_transform(traindata)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_new_counts = count_vect.transform(testdata)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#Naive Bayes model
model=MultinomialNB(fit_prior=True)
model.fit(X_train_tfidf,trainlabel)
predLabel = model.predict(X_new_tfidf)
conmatrix = confusion_matrix(testlabel, predLabel)
print("Confusion matrix:", conmatrix)
print("Accuracy Score", accuracy_score(testlabel, predLabel))
sns.heatmap(conmatrix, cmap="RdPu", annot=True, fmt="d", xticklabels=xlab, yticklabels=ylab)
plt.title("Naive Bayes Model")
plt.show()

#SVM model
sc = StandardScaler()
model = SVC(kernel = 'linear', random_state = 0)
model.fit(X_train_tfidf,trainlabel)
predLabel = model.predict(X_new_tfidf)
conmatrix = confusion_matrix(testlabel, predLabel)
print("Confusion matrix:", conmatrix)
print("Accuracy Score", accuracy_score(testlabel, predLabel))
sns.heatmap(conmatrix, cmap="RdPu",annot=True, fmt="d", xticklabels=xlab, yticklabels=ylab)
plt.title("SVM Model")
plt.show()

#Logistic Regression model
model = LogisticRegression(random_state = 0)
model.fit(X_train_tfidf,trainlabel)
predLabel = model.predict(X_new_tfidf)
conmatrix = confusion_matrix(testlabel, predLabel)
print("Confusion matrix:", conmatrix)
print("Accuracy Score", accuracy_score(testlabel, predLabel))
sns.heatmap(conmatrix, cmap="RdPu",annot=True, fmt="d", xticklabels=xlab, yticklabels=ylab)
plt.title("Logistic Regression Model")
plt.show()