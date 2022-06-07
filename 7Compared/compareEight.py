# Algorithms Comparison
from sqlite3 import Timestamp
import warnings
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
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import time
sizingFont = 15
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8]  # Plot-frame
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})  # Inside
print("========================|SISFALL_No_Headings|========================11=================")

url = "./../6dataXYZ/YSis1ALLS.csv"
dataframe = pd.read_csv(url, header=1, delimiter=",")
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
seed = 7
print(X.shape)
print(Y.shape)
print(array.shape)
seed = 1
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
# models.append(('9-Lrg-BAD', LinearRegression()))
# evaluate each model in turn
names = []
results1 = []  # Name
results2 = []  # Accuracy
results3 = []  # Times
scoring = 'accuracy'


print("======================================Model=========60==============================================")
for name, model in models:
    timeStart = time.time()
    start = time.time()
    kfold = model_selection.KFold(
        n_splits=10, shuffle=True, random_state=seed)  
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    names.append(name)
    results1.append(cv_results)    
    if (name!='6-SVM' or name!='7-RF'or name!='8-XG'):       
        timeEnd = time.time()-timeStart
        print(timeEnd,"- Finist-YES-TimeStart-1",name)
        results2.append(time.time()-timeStart)
        print(timeEnd,"- Start -YES-TimeStart-2",name)
        print('============time1=======================IIIIIIIIIIIIIIFFFFFFFFFFF=======YES==============')
        print('')
    else:
        timeEnd = time.time()-timeStart
        print(timeEnd,"- Finish-NO-TimeStart-3",name)
        results2.append(time.time()-(timeStart)/4)
        print(timeEnd,"- Start -NO-TimeStart-4",name)
        print('============time2========================EEEEEEEEEEEEEELLLLLLSSSS========NO============')
        print('')
    results3.append(cv_results.mean())
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg) 
    # your code
    stop = time.time()
    print(name,"The time of the run:", stop - start)
print("======================================Model=========86==============================================")
print('===========Result1-Names1=================')
print(names)
print('===========Result2-Times2=================')
print(results2)
print('===========Result3-Accuracy3==============')
print(results3)
print('=========================================')

print("==================|SCORE_close_X1|========================")
fig = plt.figure(figsize=(12, 8)) ## h=12 w=8
plt.grid(True)
plt.bar(names, results2, color='green', width=0.8, label='accuracy')
plt.legend(loc='best', fontsize=sizingFont)
plt.xlabel("Execution clock times in seconds", fontsize=sizingFont)
plt.ylabel("Time in milisecons", fontsize=sizingFont)
plt.xticks(rotation=30)
#plt.ylim(0, 100)
plt.title("1-Accuracy acieved spend for algorithm completing  its execution",
          fontsize=sizingFont)
plt.savefig('../UXVIEWS/others/xCompare1.png', dpi=99,
            fontsize=sizingFont, bbox_inches='tight', transparent=True)
plt.savefig('../UXviews/xCompare1.png')
plt.show()

print("==================|SCORE_close_X2|========================")

label1 = 'Time spent in completing the execution of a single task'
label2 = 'Computational costs for the compared Algorithms'
label3 = 'The period for task execution'
fig = plt.figure(figsize=(12, 8)) ## h=12 w=8
plt.grid(True)
plt.bar(names, results3, color='blue', width=0.9, label='Time')
plt.legend(loc='best', fontsize=sizingFont)
plt.title(label1, fontsize=sizingFont)
plt.xlabel(label2, fontsize=sizingFont)
plt.ylabel(label3, fontsize=sizingFont)
plt.xticks(rotation=30)
#plt.ylim(0, 100)
plt.savefig('../UXVIEWS/others/xCompare2.png', dpi=99,
            fontsize=sizingFont, bbox_inches='tight', transparent=True)
plt.savefig('../UXviews/xCompare2.png')
plt.show()

print("==================|SCORE_close_X3|========================")

fig = plt.figure(figsize=(12, 8)) ## h=12 w=8
plt.grid(True)
plt.bar(names, results2, color='blue', width=0.5, label='Accuracy')
plt.bar(names, results3, color='red', width=0.9, label='Time')
plt.legend(loc='best', fontsize=sizingFont)
plt.xlabel("Eight different algorithms and thir computation costs", fontsize=sizingFont)
plt.ylabel("Clock time and Accuracy levels", fontsize=sizingFont)
plt.title("3-Times spend for algorithm completing  its execution", fontsize=sizingFont)
plt.title('3-ACCURACY-SECOND VERTICAL', fontsize=sizingFont)
plt.xticks(rotation=30)
# plt.ylim(0, 100)
plt.savefig('../UXVIEWS/others/xCompare3.png', dpi=99,
            fontsize=sizingFont, bbox_inches='tight', transparent=True)
plt.savefig('../UXviews/xCompare3.png')
plt.show()
# plt.close()

print("==================|SCORE_THREE_X|========================")
