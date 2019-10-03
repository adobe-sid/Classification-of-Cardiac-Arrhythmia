import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from drop_highlycorelated import clf,xtrain,ytrain,xtest,ytest,X_important_train,X_important_test
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_important_train, ytrain)

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
        cv_results = cross_val_score(logistic_regression_model,X_important_train,ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        #names.append(name)    
        print(results)
        
y_pred = clf.predict(xtest)

# View The Accuracy Of Our Full Feature (4 Features) Model
print("Accuracy of full features : ",end=" ")
print(accuracy_score(ytest, y_pred))

y_important_pred = logistic_regression_model.predict(X_important_test)

print("Accuracy of limited features : ",end=" ")
print(accuracy_score(ytest, y_important_pred))

print()
print()

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
 
