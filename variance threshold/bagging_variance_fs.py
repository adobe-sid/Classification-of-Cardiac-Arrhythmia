import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file as load_svm
from sklearn.model_selection import KFold
from variance_threshold import clf,xtrain,ytrain,xtest,ytest,X_important_train,X_important_test
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

clf_imp=BaggingClassifier(knn(n_neighbors=2,p=2,metric='minkowski'))
clf_imp.fit(X_important_train,ytrain)

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
        cv_results = cross_val_score(clf_imp,X_important_train,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)
# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(xtest)

# View The Accuracy Of Our Full Feature (4 Features) Model
print("Accuracy of full features : ",end=" ")
print(accuracy_score(ytest, y_pred))

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_imp.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
print("Accuracy of limited features : ",end=" ")
print(accuracy_score(ytest, y_important_pred))

print()
print()
##print("training accuracy: {}".format(100*clf.score(xtrain,ytrain)))
##print("testing accuracy: {}".format(100*clf.score(xtest,ytest)))

y_true=ytest
y_pred=clf.predict(xtest)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("For full feature dataset...")
print()
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()
print()

accuracy_imp = accuracy_score(y_true, y_important_pred)
precision_imp, recall_imp, f1_score_imp, _ = precision_recall_fscore_support(y_true, y_important_pred, average='micro')

print("For limites feature dataset...")
print()
print("Accuracy: ", accuracy_imp)
print("Precision: ", precision_imp)
print("Recall: ", recall_imp)
print("F1 score: ", f1_score_imp)
