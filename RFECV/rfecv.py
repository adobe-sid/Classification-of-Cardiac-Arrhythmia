import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from vecstack import stacking
from sklearn.metrics import mean_absolute_error

df=pd.read_csv("arrhythmia_data.csv", ',', na_values=['?'])
#print(df.mean())
a = df.fillna(df.mean())
#print(a)
ds=np.array(a)
#print(df.shape)
X=np.array(a.drop(['8'],1))
y=np.array(a['8'])

 
from sklearn import linear_model
ols = linear_model.LinearRegression()
rfecv = RFECV(estimator=ols, step=1, scoring='neg_mean_squared_error')

# Fit recursive feature eliminator 
rfecv.fit(X, y)

# Recursive feature elimination
rfecv.transform(X)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=43)

#1
clf=knn(n_neighbors=10, p=2, algorithm='auto', leaf_size=30, metric='minkowski')
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For knn...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


##2''' BAGGING CLASSIFIER'''
clf=BaggingClassifier(knn(n_neighbors=2,p=2,metric='minkowski'))
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For bagging knn...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()

clf=BaggingClassifier(svm.SVC(kernel='linear',C=1).fit(xtrain,ytrain))
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For bagging svm...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()

clf=BaggingClassifier(RandomForestClassifier(n_estimators=1000,random_state=42))
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For bagging rf...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#3
clf=BernoulliNB()
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For naivebias...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#4
clf=svm.SVC(kernel='rbf',C=100,gamma=0.04)
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For svm...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#5
clf=tree.DecisionTreeClassifier(random_state=17)
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For dtree...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#6
clf=AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=1))
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For Adaboost...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()

#7
clf=GradientBoostingClassifier(learning_rate=0.01,random_state=1)
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For Gradiantboost...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#8
clf=LogisticRegression()
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For Logistic regression...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#9
clf=RandomForestClassifier(n_estimators=1000,random_state=62)
clf.fit(xtrain,ytrain)

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For rf...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()


#10
models = [
    svm.SVC(kernel='linear',C=1),
        
    RandomForestClassifier(random_state=42, n_jobs=-1, 
                          n_estimators=1000, max_depth=3),
    BaggingClassifier(svm.SVC(kernel='linear',C=1))
]

S_train, S_test = stacking(models,                     # list of models
                           xtrain, ytrain, xtest,      # data,            # regression task (if you need 
                                                       #     classification - set to False)
                           mode='oof_pred_bag',        # mode: oof for train set, predict test 
                           regression=True,                     #     set in each fold and find mean
                           save_dir=None,              # do not save result and log (to save 
                                                       #     in current dir - set to '.')
                           metric=mean_absolute_error, # metric: callable
                           n_folds=4,                  # number of folds
                           shuffle=True,               # shuffle the data
                           random_state=0,             # ensure reproducibility
                           verbose=2)

clf_imp=GradientBoostingClassifier(learning_rate=0.01,random_state=1)
clf_imp.fit(xtrain,ytrain)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(xtest)

# View The Accuracy Of Our Full Feature (4 Features) Model
print("Accuracy of full features : ",end=" ")
print(accuracy_score(ytest, y_pred))

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_imp.predict(xtest)

print()
print()

#kfold
from sklearn.model_selection import cross_val_score,KFold
n_folds = []
n_folds.append(('K2', 2))
n_folds.append(('K4', 4))
n_folds.append(('K5', 5))
n_folds.append(('K10', 10))

seed = 7

for name, n_split in n_folds:
        results = []
        names = []
        print(name)  
        kfold = KFold(
        n_splits=n_split, random_state=seed)
        cv_results = cross_val_score(clf_imp,xtrain,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)
        
y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For stacking...")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()
print()
