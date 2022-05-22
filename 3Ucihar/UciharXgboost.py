from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
# from DecisionTreeClassifier  # 3
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import xgboost as xgb
import xgboost
#from sklearn import tree
import pickle
import warnings
warnings.simplefilter('ignore')
epochs = 2022
size = 15
plt.rc('axes', titlesize=20)
plt.rc('font', size=15)
plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the y tick labels
plt.rc('legend', fontsize=10)  # fontsize of the legend
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams["figure.autolayout"] = True

print("==================================|Dataset_Used|====================================1============")

datasetName = './../5dataXYZ/ucihar1.csv'
df2 = pd.read_csv(datasetName, header=0, delimiter=",")
print(df2.head(5))

df = pd.DataFrame(df2)
df2 = df2.apply(pd.to_numeric)
corrs = df2.corr()['Activity'].abs()

print('countedactivities1->', corrs)
print('countedactivities2->', corrs)
print('countedactivities3->', corrs)
print('countedactivities4->', corrs)
columns = corrs[corrs > .01].index
corrs = corrs.filter(columns)
corrs
print(corrs)
X = df2.iloc[:, 1:19]
Y = df2.iloc[:, 0:1]
print(df.head(5))
print("=====================================|HeadersOnly|==================================2=============")
headers = list(X)
print(headers)
print("====================================|HeadersAndData|================================3=============")
print(X)
print(df.shape)
print("===================================|TrainingAndTest|================================4=============")

# Initialising the XGBoost machine learning model
train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.33, stratify=Y, random_state=5)
print(train_X.shape, test_X.shape)
print()
print('Number of rows in Train dataset: {train_df.shape[0]}')
print(train_Y['Activity'].value_counts())
print()
print('Number of rows in Test dataset: {test_df.shape[0]}')
print(test_Y['Activity'].value_counts())

model = xgb.XGBClassifier(max_depth=12,  # 99,
                          subsample=0.33,
                          # objective='multi:softmax',
                          binary='logistic',
                          n_estimators=epochs,
                          learning_rate=0.01,
                          eval_metric='logloss',
                          verbosity=0)
eval_set = [(train_X, train_Y), (test_X, test_Y)]
model.fit(train_X, train_Y.values.ravel(), early_stopping_rounds=epochs,
          eval_metric=["error", "logloss", "auc"], eval_set=eval_set,  verbose=True)

print("===============================|Accuracy|============================================5==============")

# make predictions for test data
y_pred = model.predict(test_X)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(test_Y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# save model to file
pickle.dump(model, open("x1_model.pkl", "wb"))

print("=================================|Scores|=|Importance|=============================6==============")

# - cross validataion
scores = cross_val_score(model, train_X, train_Y, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model, train_X, train_Y, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
ypred = model.predict(test_X)
cm = confusion_matrix(test_Y, ypred)
# print(cm)

# Feature importance-# Plot the top 7 features
xgboost.plot_importance(model, max_num_features=10)
plt.grid()
plt.title('83.C1-XGBoost Features(9) Importance | '+datasetName)
plt.title('XGBoost Features(12) Importance | '+datasetName)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/83C1.png')
plt.savefig('../UXviews/THREE/83C1.png')
plt.show()
# Predict the trading signal on test datset
y_pred = model.predict(test_X)
# Get the classification report
# print(classification_report(test_Y, y_pred))

# Confusion Matrix
print("====================================|Confussion_Matrix|=================================7============")
# plt.rc('axes', titlesize=22)
# plt.rc('font', size=25)
# plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=18)  # fontsize of the x tick labels
# plt.rc('ytick', labelsize=18)  # fontsize of the y tick labels
# plt.rc('legend', fontsize=15)  # fontsize of the legend
# plt.rcParams['figure.figsize'] = [10, 10]
# plt.rcParams["figure.autolayout"] = True
# #plt.imshow(X, aspect='auto')

print(test_Y.shape)
print(y_pred.shape)
array = confusion_matrix(y_pred,test_Y)
print(array)
print(array.shape)
array = confusion_matrix(test_Y, y_pred)
print(array.shape)
df = pd.DataFrame(array, index=['ADLs (others)', 'LAYING'], columns=[
                  'ADLs (others)', 'LAYING'])
plt.figure(figsize=(12, 6))
sn.heatmap(df, annot=True, cmap='Blues', fmt='g')

plt.xlabel('Predicted', fontsize=size,)
plt.ylabel('Actual')
plt.suptitle('XGBoost Confusion Matrix of Test Dataset | '+datasetName)
plt.suptitle('Confusion matrix - Ucihar dataset')
plt.tight_layout()
plt.savefig('../UXviews/83C2.png', aspect='auto')
plt.savefig('../UXviews/THREE/83C2.png', aspect='auto')
plt.show()

plt.rc('axes', titlesize=20)
plt.rc('font', size=15)
plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the y tick labels
plt.rc('legend', fontsize=10)  # fontsize of the legend
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams["figure.autolayout"] = True

print("======================================|Three_Plots|=====================================8============")
# Predict and Classification report-# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
a1 = round(results['validation_0']['auc'][1], 4)
a2 = round(results['validation_1']['auc'][1], 4)
# print(a1)
# print(a2)
# print(results)
print("==================|=====|ONE|==================")

# 1 plot classification auc
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['validation_0']['auc'],
        label='Train-%'+str(round(a1*100, 4)))
ax.plot(x_axis, results['validation_1']['auc'],
        label='Test- %'+str(round(a2*100, 4)))
ax.legend(fontsize=size, loc="best")
plt.ylabel('AUC-Classification')
# plt.axis([0, epochs, 0.0000, 1.2000])
plt.title('83.C3-AUC-XGBoost Classification Accuracy | '+datasetName)
plt.title('XGBoost Area Under The Curve(AUC) Accuracy | '+datasetName)
plt.grid()
plt.tight_layout()
plt.savefig('../UXviews/83C3.png')
plt.savefig('../UXviews/THREE/83C3.png')
plt.show()

# 2 plot log loss
print("==================|====TWO====================")
a1 = results['validation_0']['logloss'][1]
a2 = results['validation_1']['logloss'][1]
# print(a1)
# print(a2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['validation_0']['logloss'],
        label='Train- %'+str(round(a1*100, 4)))
ax.plot(x_axis, results['validation_1']['logloss'],
        label='Test-  %'+str(round(a2*100, 4)))
ax.legend(fontsize=size, loc="best")
plt.ylabel('Log Loss')
#plt.axis([0, epochs, 0.0, 1.1])
plt.title('83.C4-XGBoost LogLoss | '+datasetName)
plt.title('XGBoost LogLoss | '+datasetName)
plt.grid()
plt.tight_layout()
plt.savefig('../UXviews/83C4.png')
plt.savefig('../UXviews/THREE/83C4.png')
plt.show()

# 3 plot classification error
print("==================|====THREE=====================")
a1 = results['validation_0']['error'][1]
a2 = results['validation_1']['error'][1]
# print(a1)
# print(a2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['validation_0']['error'],
        label='Train- %'+str(round(a1*100, 4)))
ax.plot(x_axis, results['validation_1']['error'],
        label='Test-  %'+str(round(a2*100, 4)))
ax.legend(fontsize=size, loc="best")
plt.ylabel('Classification Error')
plt.title('83.C5-XGBoost Classification Error | '+datasetName)
plt.title('XGBoost Classification Error | '+datasetName)
plt.grid()
plt.tight_layout()
plt.savefig('../UXviews/83C5.png')
plt.savefig('../UXviews/THREE/83C5.png')
plt.show()


print("=====================================|DescisionTreeClassiferMAP|=======================9============")

# clf = DecisionTreeClassifier(max_depth=20)
# x_train, x_test, y_train, y_test = train_test_split(X, Y)
# fig = clf.fit(x_train, y_train)

# plt.figure(figsize=(12, 6))
# tree.plot_tree(fig, fontsize=7)
# plt.title('83.C6-AUC-XGBoost Decision Tree Map | '+datasetName)
# plt.title('XGBoost Decision Tree Map | '+datasetName)
# plt.tight_layout()
# plt.tight_layout()
# plt.savefig('../UXviews/83C6.png')
# plt.savefig('../UXviews/THREE/83C6.png')
# plt.show()

print("==============================|Successfully_Completed!|========================|THREE_CC|=====|CCC|==================")
