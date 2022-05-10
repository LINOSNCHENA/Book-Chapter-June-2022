from io import StringIO
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
plt.bar(names2, results3, color='red', width=0.5, label='Accuracy')
plt.legend(loc='best')
plt.xlabel("Machine learning Algorithms and their time consumption costs")
plt.ylabel("Time consumed with resultant accuracy levels")
plt.title("Time as a cost and accuracy achieved")
plt.grid(True)
plt.savefig('../UXviews/table4/T5a.png')
plt.show()
print('============END================')

fig = plt.figure(figsize=(14, 10))
plt.bar(names2, results3, color='gray', width=0.5, label='Accuracy')
plt.legend(loc='best')
plt.xlabel("Analysing the accuracy computation cost of the different algorithms")
plt.ylabel("Clock time and Accuracy levels")
plt.grid(True)
plt.title("Algorithms and their Accuracies compared")
plt.savefig('../UXviews/table4/T5b.png')
plt.show()
print('============END================')


fig, ax1 = plt.subplots(figsize=(14,10))
# fig = plt.figure(figsize=(14, 10))
plt.bar(names2, results2, color='gray', width=0.5, label='Time')
plt.legend(loc='best')
plt.xlabel("Analysing the time computation cost of the different algorithms")
plt.ylabel("Clock time and Accuracy levels")
plt.grid(True)
plt.title("Algorithms and their task completion time period")
plt.savefig('../UXviews/table4/T5c.png')
plt.show()
print('============END========128========')

s = StringIO("""     amount     price
A     40929   4066443
B     93904   9611272
C    188349  19360005
D    248438  24335536
E    205622  18888604
F    140173  12580900
G     76243   6751731
H     36859   3418329
I     29304   2758928
J     39768   3201269
K     30350   2867059""")

df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)

fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.4
df.amount.plot(kind='bar', color='red', ax=ax, width=width, position=1)
df.price.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
print(df)
ax.set_ylabel('Amount')
ax2.set_ylabel('Price')
plt.show()
print('============END========128========')
df = pd.DataFrame(results3,index=names[:,0]),
df2 = pd.DataFrame(results2,index=names[:,0]),

# creating the dataframe
df4 = pd.DataFrame(data = array, 
                  index = names, 
                  columns = results2)
print(df)
print(df2)
df5 = df.append(names, ignore_index = True)
df25 = df2.append(names, ignore_index = True)
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.4
df.plot(names,kind='bar', color='red', ax=ax, width=width, position=1)
df2.plot(names,kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.set_ylabel('Accuray')
ax2.set_ylabel('Time')
plt.show()
# Display

fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.4
df.plot(names,kind='bar', color='red', ax=ax, width=width, position=1)
df2.plot(names,kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.set_ylabel('Accuray')
ax2.set_ylabel('Time')
plt.show()


