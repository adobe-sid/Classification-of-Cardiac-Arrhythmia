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

clf_imp=CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
clf_imp.fit(X_important_train,ytrain,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(X_important_test, ytest))
clf_imp.fit(X_important_train,ytrain)

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
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print("For full feature dataset...")
print()
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
print()
print()

accuracy_imp = accuracy_score(y_true, y_important_pred)
precision_imp, recall_imp, f1_score_imp, _ = precision_recall_fscore_support(y_true, y_important_pred, average='binary')

print("For limites feature dataset...")
print()
print("Accuracy: ", accuracy_imp)
print("Precision: ", precision_imp)
print("Recall: ", recall_imp)
print("F1 score: ", f1_score_imp)
