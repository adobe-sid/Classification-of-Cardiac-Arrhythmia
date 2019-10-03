import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file as load_svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import tree
import lightgbm as lgb

df=pd.read_csv(r'C:\Users\user\Desktop\MY\permission.csv',";")

ds=np.array(df)
ds.shape

x=ds[:,:-1]
y=ds[:,-1]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

model=CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(xtrain,ytrain,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(xtest, ytest))
print("testing accuracy for : {}".format(100*model.score(xtest,ytest)))
print()
train_data=lgb.Dataset(xtrain,label=ytrain)
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 
##y_pred=model.predict(xtest)
##for i in range(0,185):
##   if y_pred[i]>=0.5: 
##   y_pred[i]=1
##else: 
##   y_pred[i]=0


##print("training accuracy: {}".format(100*clf.score(xtrain,ytrain)))
##print("testing accuracy for dtree: {}".format(100*model.score(xtest,ytest)))
print()

