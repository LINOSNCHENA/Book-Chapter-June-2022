# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
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
import warnings
warnings.simplefilter('ignore')
sizingFont = 15
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8]  # Plot-frame
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})  # Inside

print("========================|SISFALL_No_Headings|========================|FIRST|=================")

dataframe = pd.read_csv('./../6dataXYZ/YSis1ALLS.csv', header=1, delimiter=",")
print(dataframe.shape)
l = -1*len(dataframe.columns)
print(l)
print(dataframe.head(4))
dataframe = dataframe.iloc[0:19090:, (l):]  # 22-Records, last-15 headers
array = dataframe.values
X = array[:, 1:19]
Y = array[:, 0:1]
# prepare configuration for cross validation test harness

seed = 7
print(X.shape)
print(Y.shape)
print(array.shape)
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
results = []
names = []
scoring = 'accuracy'
seed = 10

print("=========================|Model|============================================")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
df1 = pd.DataFrame()
cols_swap1 = results[0][0].round(10)
cols_swap1x = results[0][1].round(10)
cols_swap1z = results[0][2].round(10)
print(cols_swap1, cols_swap1x, cols_swap1z)

print("======================|Make-Decimal|============================================")
cols_swap1 = results[0].round(10)
cols_swap2 = results[1].round(10)
cols_swap3 = results[2].round(10)
cols_swap4 = results[3].round(10)
cols_swap5 = results[4].round(10)
cols_swap6 = results[5].round(10)
cols_swap7 = results[6].round(10)
cols_swap8 = results[7].round(10)
df1.insert(0, 'ACC1_LR_SISFALL', cols_swap1, True)
df1.insert(1, 'ACC2_LDA_SISFALL', cols_swap2, True)
df1.insert(2, 'ACC3_KNN_SISFALL', cols_swap3, True)
df1.insert(3, 'ACC4_DTC_SISFALL', cols_swap4, True)
df1.insert(4, 'ACC5_GNB_SISFALL', cols_swap5, True)
df1.insert(5, 'ACC6_SVM_SISFALL', cols_swap6, True)
df1.insert(6, 'ACC7_RF_SISFALL', cols_swap7, True)
df1.insert(7, 'ACC8_XG_SISFALL', cols_swap8, True)
print(df1)

df1.to_csv(r'ACC/ACCURACYSISFALL.csv', index=0)

print("========================|Testing|==================================================")
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Algorithm Comparison - Test- Sisfall-Dataset')
plt.title('1-Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.axhline(y=1.05, xmin=0.1, xmax=0.9, dashes=(5, 5), c='b')
plt.axhline(y=0.65, xmin=0.06, dashes=(5, 5), c='r')
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/V1.png')
plt.savefig('../UXviews/3ACC/V1.png')

print(type(results))
y_test = array[8:9:, 1:19]
y_pred = array[18:19:, 1:19]
print(y_test.shape)
print(y_pred.shape)

print("==============================|MOBIACT_STARTS_Without_Headers|==============|SECOND|===============")

dataframe = pd.read_csv('./../6dataXYZ/XMobiACT7.csv', header=1, delimiter=",")
print((dataframe.shape[0])/3)
print(dataframe.head(4))
l = -1*len(dataframe.columns)
print(l)
dataframe = dataframe.iloc[0:19090:, l:]  # 22-Records, last-15 headers
array = dataframe.values
X = array[:, 1:19]
Y = array[:, 0]
# prepare configuration for cross validation test harness

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
results = []
names = []
scoring = 'accuracy'
seed = 10

print("========================================================22=================")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
df1 = pd.DataFrame()
cols_swap1 = results[0][0].round(10)
cols_swap1x = results[0][1].round(10)
cols_swap1z = results[0][2].round(10)
print(cols_swap1, cols_swap1x, cols_swap1z)

print("==========================|Make-Decimal|=======================================")
cols_swap1 = results[0].round(10)
cols_swap2 = results[1].round(10)
cols_swap3 = results[2].round(10)
cols_swap4 = results[3].round(10)
cols_swap5 = results[4].round(10)
cols_swap6 = results[5].round(10)
cols_swap7 = results[6].round(10)
cols_swap8 = results[7].round(10)
df1.insert(0, 'ACC1_LR_MOBIACT', cols_swap1, True)
df1.insert(1, 'ACC2_LDA_MOBIACT', cols_swap2, True)
df1.insert(2, 'ACC3_KNN_MOBIACT', cols_swap3, True)
df1.insert(3, 'ACC4_DTC_MOBIACT', cols_swap4, True)
df1.insert(4, 'ACC5_GNB_MOBIACT', cols_swap5, True)
df1.insert(5, 'ACC6_SVM_MOBIACT', cols_swap6, True)
df1.insert(6, 'ACC7_RF_MOBIACT', cols_swap7, True)
df1.insert(7, 'ACC8_XG_MOBIACT', cols_swap8, True)
print(df1)
df1.to_csv(r'ACC/ACCURACYMOBIACT.csv', index=0)

print("==============================|Testing|============================================")
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Algorithm Comparison - Test- MobiAct-Dataset')
plt.title('2-Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.axhline(y=1.05, xmin=0.1, xmax=0.9, dashes=(5, 5), c='b')
plt.axhline(y=0.65, xmin=0.06, dashes=(5, 5), c='r')
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/V2.png')
plt.savefig('../UXviews/3ACC/V2.png')

print(type(results))
y_test = array[8:9:, 1:19]
y_pred = array[18:19:, 1:19]
print(y_test.shape)
print(y_pred.shape)

print("============================|UCIHAR_STARTS_Without_Head|==============|THIRD|=================")

dataframe = pd.read_csv('./../6dataxyz/Ucihar2.csv', header=1, delimiter=",")
print(dataframe.shape)
l = -1*len(dataframe.columns)
print(l)
print(dataframe.head(4))
dataframe = dataframe.iloc[0:19090:, (l):]  # 22-Records, last-15 headers
array = dataframe.values
X = array[:, 1:19]
Y = array[:, 0:1]

# prepare configuration for cross validation test harness
seed = 7
print(X.shape)
print(Y.shape)
print(array.shape)
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
results = []
names = []
scoring = 'accuracy'
seed = 10

print("======================|Model|===============================================")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
df1 = pd.DataFrame()
cols_swap1 = results[0][0].round(10)
cols_swap1x = results[0][1].round(10)
cols_swap1z = results[0][2].round(10)
print(cols_swap1, cols_swap1x, cols_swap1z)

print("====================|Make-Decimal|==========================================")
cols_swap1 = results[0].round(10)
cols_swap2 = results[1].round(10)
cols_swap3 = results[2].round(10)
cols_swap4 = results[3].round(10)
cols_swap5 = results[4].round(10)
cols_swap6 = results[5].round(10)
cols_swap7 = results[6].round(10)
cols_swap8 = results[7].round(10)

df1.insert(0, 'ACC1_LR_UCIHAR', cols_swap1, True)
df1.insert(1, 'ACC2_LDA_UCIHAR', cols_swap2, True)
df1.insert(2, 'ACC3_KNN_UCIHAR', cols_swap3, True)
df1.insert(3, 'ACC4_DTC_UCIHAR', cols_swap4, True)
df1.insert(4, 'ACC5_GNB_UCIHAR', cols_swap5, True)
df1.insert(5, 'ACC6_SVM_UCIHAR', cols_swap6, True)
df1.insert(6, 'ACC7_RF_UCIHAR', cols_swap7, True)
df1.insert(7, 'ACC8_XG_UCIHAR', cols_swap8, True)
print(df1)
df1.to_csv(r'ACC/ACCURACYUCIHAR.csv', index=0)

print("===========================|Testing|========================================")
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Algorithm Comparison - Test- Ucihar-Dataset')
plt.title('3-Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.axhline(y=1.05, xmin=0.1, xmax=0.9, dashes=(5, 5), c='b')
plt.axhline(y=0.65, xmin=0.06, dashes=(5, 5), c='r')
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/V3.png')
plt.savefig('../UXviews/3ACC/V3.png')

print(type(results))
y_test = array[8:9:, 1:19]
y_pred = array[18:19:, 1:19]
print(y_test.shape)
print(y_pred.shape)

print("=================================|S1_SUCCESSFULLY_COMPLETED|==============|VERTICAL|=================")