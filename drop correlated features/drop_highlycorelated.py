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

df=pd.read_csv("arrhythmia_data.csv", ',', na_values=["?"])
#print(df.mean())
a = df.fillna(df.mean())
#print(a)
ds=np.array(a)
#print(df.shape)
x=np.array(a.drop(['8'],1))
y=np.array(a['8'])
count=0
s = df.ix[1,:-1].index
labels=[]
labels=s
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(xtrain,ytrain)
i=0
for feature in zip(labels, clf.feature_importances_):
##    if(feature[1]>0.001):
##        count+=1
##        print(labels[i])
        i+=1
##        print(feature)
##print(count)

# Create correlation matrix
x=pd.DataFrame(x)
corr_matrix = x.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than this
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
##sfm = SelectFromModel(clf, threshold=0.001)
##sfm.fit(xtrain, ytrain)
print(to_drop)
##for feature_list_index in sfm.get_support(indices=True):
##        print(labels[feature_list_index])

x.drop(x.columns[to_drop], axis=1)
X_important_train,X_important_test,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

##X_important_train = sfm.transform(xtrain)
##X_important_test = sfm.transform(xtest)

##clf_important =svm.SVC(kernel='linear',C=1).fit(xtrain,ytrain)
##
### Train the new classifier on the new dataset containing the most important features
##clf_important.fit(X_important_train, ytrain)
##y_pred = clf.predict(xtest)
##
### View The Accuracy Of Our Full Feature (4 Features) Model
##print(accuracy_score(ytest, y_pred))
##
##y_important_pred = clf_important.predict(X_important_test)
### View The Accuracy Of Our Limited Feature (2 Features) Model
##print(accuracy_score(ytest, y_important_pred))
##
