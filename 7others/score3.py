
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Compare Algorithms
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
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
# prepare configuration for cross validation test harness
import warnings
warnings.simplefilter('ignore')
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
# prepare configuration for cross validation test harness
print("===========================================================12=================")
import time
seed = 7
print(X.shape)
print(Y.shape)
print(array.shape)
seed = 7
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
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    x = time.time()
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    results2.append(time.time()-x)
    results3.append(cv_results.mean())
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
print('============================')
print(names)
print('============================')
print(results2)
print(results3)
print('============================')

# Function to add value labels


def valuelabel(results2, results3):
    for i in range(len(results2)):
        plt.text(i, results3[i], results3[i], ha='center',
                 bbox=dict(facecolor='cyan', alpha=0.8))
# fig = plt.figure(figsize=(14, 10))
# plt.bar(names, results2, color='gray', width=0.8, label='accuracy')
# # valuelabel(names, results2)
# # plt.bar(names, results2, color ='blue',width = 0.8,label='time')
# # plt.bar(names,results2, color ='green',width = 0.8,label='time1')
# # plt.bar(names,results3, color ='yellow',width = 0.8,label='time2')
# # Call function
# # valuelabel(results2, results3)
# plt.legend(loc='best')
# plt.xlabel("Execution clock times seconds")
# plt.ylabel("Time in milisecons")
# plt.title("Times spend for algorithm completing  its execution")
# plt.show()


fig = plt.figure(figsize=(14, 10))
plt.bar(names, results2, color='blue', width=0.9, label='Time')
plt.bar(names, results3, color='gray', width=0.5, label='Accuracy')
plt.legend(loc='best')
plt.xlabel("Eight different algorithms and thir computation costs")
plt.ylabel("Clock time and Accuracy levels")
plt.title("Times spend for algorithm completing  its execution")
plt.show()
