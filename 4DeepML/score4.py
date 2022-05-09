import pandas
import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import time
# load dataset
print("========================|SISFALL_No_Headings|========================11=================")

dataframe = pd.read_csv('./../5dataXYZ/YSis1ALLS.csv', header=1, delimiter=",")
print(dataframe.shape)
l = -1*len(dataframe.columns)
print(l)
print(dataframe.head(4))
dataframe = dataframe.iloc[0:19090:, (l):]  # 22-Records, last-15 headers
array = dataframe.values
X = array[:, 1:19]
Y = array[:, 0:1]
# Y=Y.numpy.squeeze()
# X=X.numpy.squeeze()
# prepare configuration for cross validation test harness
print("===========================================================12=================")
import time
print(X.shape)
print(Y.shape)
print(array.shape)
seed = 3
# prepare models
models = []

models.append(('1-LR', LogisticRegression()))
models.append(('2-LDA', LinearDiscriminantAnalysis()))
models.append(('3-KNN', KNeighborsClassifier()))
models.append(('4-DTC', DecisionTreeClassifier()))
models.append(('5-GNB', GaussianNB()))
models.append(('6-SVM', SVC()))
models.append(('7-RF', RandomForestClassifier()))
models.append(('8-XG', XGBClassifier(eval_metric='logloss')))
# evaluate each model in turn
# prepare models

results = []
names = []
names2 = []
results2 = []
results3 = []
results4 = []
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    x = time.time()
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    names2.append(name)
    results2.append(time.time()-x)
    results3.append(cv_results.mean())
    results4.append(name)
    results4.append(time.time()-x)
    results4.append(cv_results.mean())
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
print('============================')
print(names)
print('============================')
print(results2)
print(results3)
print('==============80==============')
fig = plt.figure(figsize=(14, 10))
plt.bar(names2, results2, color='blue', width=0.9, label='Time')
plt.bar(names2, results3, color='gray', width=0.5, label='Accuracy')
plt.legend(loc='best')
plt.xlabel("Eight different algorithms and thir computation costs")
plt.ylabel("Clock time and Accuracy levels")
plt.title("Times spend for algorithm completing  its execution")
plt.show()
print('============END================')

fig = plt.figure(figsize=(14, 10))
#plt.bar(names2, results2, color='blue', width=0.9, label='Time')
plt.bar(names2, results3, color='gray', width=0.5, label='Accuracy')
plt.legend(loc='best')
plt.xlabel("Eight different algorithms and thir computation costs")
plt.ylabel("Clock time and Accuracy levels")
plt.title("5a-Times spend for algorithm completing  its execution")
plt.savefig('../UXviews/table4/T5a.png')
plt.show()
print('============END================')


fig = plt.figure(figsize=(14, 10))
#plt.bar(names2, results2, color='blue', width=0.9, label='Time')
plt.bar(names2, results3, color='gray', width=0.5, label='Accuracy')
plt.legend(loc='best')
plt.xlabel("Eight different algorithms and thir computation costs")
plt.ylabel("Clock time and Accuracy levels")
plt.title("5b-Times spend for algorithm completing  its execution")
plt.savefig('../UXviews/table4/T5b.png')
plt.show()
print('============END================')

