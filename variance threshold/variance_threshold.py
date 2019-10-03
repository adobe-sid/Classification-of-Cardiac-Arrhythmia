import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

df=pd.read_csv("arrhythmia_data.csv", ',', na_values=["?"])
#print(df.mean())
a = df.fillna(df.mean())
#print(a)
ds=np.array(a)
#print(df.shape)
x=np.array(a.drop(['8'],1))
y=np.array(a['8'])
labels=[]
s = df.ix[1,:-1].index
count=0
labels=s
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(xtrain,ytrain)
i=0
for feature in zip(labels, clf.feature_importances_):
##        if(feature[1]>0.001):
##                count+=1
##        print(labels[i])
        i+=1
##        print(feature)
##print(count,i)

thresholder = VarianceThreshold(threshold=0.02)
thresholder.fit(xtrain, ytrain)
count=0
for feature_list_index in thresholder.get_support(indices=True):
    count+=1
print(count)
##        print(labels[feature_list_index])

X_important_train = thresholder.transform(xtrain)
X_important_test = thresholder.transform(xtest)
