from turtle import color
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import xgboost as xgb
import xgboost
import pickle
import warnings
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8] ## Plot-frame
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})   ## Inside
sizingFont = 15                          ## SupTitle & Xlabel
n_features = 10
seed = 1
batch_size = 1
epochs = 2022
plotMethod = "Machine-Learning"
datasetName = "Sisfall-Dataset"
urlDataset = './../6dataXYZ/YSis1ALLS.csv'
print("==================================|Dataset_Used|====================================1============")


df2 = pd.read_csv(urlDataset, header=1, delimiter=",")
print(df2.head(5))
print(df2.shape)
df2.columns = ['class_var', 'X1',	'X2',	'X3',
               'Y1',	'Y2',	'Y3',	'Z1', "Z2",	'Z3']
df = pd.DataFrame(df2)
df2 = df2.apply(pd.to_numeric)
corrs = df2.corr()['class_var'].abs()
columns = corrs[corrs > .01].index
corrs = corrs.filter(columns)
corrs
# print(corrs)
X = df2.iloc[:, 1:19]
Y = df2.iloc[:, 0:1]
# print(df.head(5))
print("=====================================|HeadersOnly|==================================2=============")
headers = list(X)
print(headers)
print("====================================|HeadersAndData|================================3=============")
print(X)
print("===================================|TrainingAndTest|================================4=============")
# Initialising the XGBoost machine learning model
train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.33, stratify=Y, random_state=5)
print(train_X.shape, test_X.shape)
print()
print('Number of rows in Train dataset: {train_df.shape[0]}')
print(train_Y['class_var'].value_counts())
print()
print('Number of rows in Test dataset: {test_df.shape[0]}')
print(test_Y['class_var'].value_counts())

model = xgb.XGBClassifier(max_depth=12,
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
pickle.dump(model, open("modelSis.pkl", "wb"))
print("=================================|Plot1-Scores|=|Importance|=============================6==============")

# - cross validataion
scores = cross_val_score(model, train_X, train_Y, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model, train_X, train_Y, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
ypred = model.predict(test_X)
cm = confusion_matrix(test_Y, ypred)
print(cm)

# Feature importance-  Plot the top 7 features
xgboost.plot_importance(model, max_num_features=21)
plt.suptitle('1-SuperTitle XGBoost Features(1) Importance | '+plotMethod, fontsize=sizingFont+4)
plt.title('XGBoost Features(2) Importance | '+plotMethod, fontsize=sizingFont)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/A1.png')
plt.savefig('../UXviews/1machLearn/A1.png')
plt.show()
# Predict the trading signal on test datset
y_pred = model.predict(test_X)
# Get the classification report
# print(classification_report(test_Y, y_pred))

# Confusion Matrix
print("====================================|Confussion_Matrix|=================================7============")
print(test_Y.shape)
print(y_pred.shape)
array = confusion_matrix(test_Y, y_pred)
df = pd.DataFrame(array, index=['ADLs', 'FALL'], columns=['ADLs', 'FALL'])
plt.figure(figsize=(12, 8))
sn.heatmap(df, annot=True, cmap='Greens', fmt='g')
plt.xlabel('Predicted values', fontsize=sizingFont)
plt.ylabel('Actual value', fontsize=sizingFont)
plt.xticks(rotation=40, fontsize=sizingFont)
plt.yticks(rotation=40, fontsize=sizingFont)
plt.suptitle('2-Xgboost Confusion Matrix | '+datasetName, fontsize=sizingFont+2)
plt.tight_layout()
plt.grid(True)
plt.savefig('../UXviews/A2.png')
plt.savefig('../UXviews/1machLearn/A2.png')
plt.show()
print("======================================|Three_Plots|=====================================8============")

# Predict and Classification report # retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
print(results)
a1 = round(results['validation_0']['auc'][2], 4)
a2 = round(results['validation_1']['auc'][2], 4)
# plot classification auc
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_axis, results['validation_0']['auc'],
        label='Train-%'+str(round(a1*100, 4)))
ax.plot(x_axis, results['validation_1']['auc'],
        label='Test- %'+str(round(a2*100, 4)))
ax.legend(loc="best",fontsize=sizingFont)
plt.ylabel('Accuracy',fontsize=sizingFont)
# plt.axis([0, epochs, 0.0000, 1.2000])
plt.title('3-XGBoost Area Under The Curve(AUC) Accuracy | ' +
          datasetName, fontsize=sizingFont)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/A3.png')
plt.savefig('../UXviews/1machLearn/A3.png')
plt.show()

# plot log loss
a1 = round(results['validation_0']['logloss'][2], 4)
a2 = round(results['validation_1']['logloss'][2], 4)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_axis, results['validation_0']['logloss'],
        label='Train-%'+str(round(a1*100, 4)))
ax.plot(x_axis, results['validation_1']['logloss'],
        label='Test- %'+str(round(a2*100, 4)))
ax.legend(fontsize=sizingFont, loc="best")
plt.ylabel('Loss in classification',fontsize=sizingFont)
#plt.axis([0, epochs, 0.0, 1.1])
plt.title('4-XGBoost LogLoss | '+datasetName, fontsize=sizingFont)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/A4.png')
plt.savefig('../UXviews/1machLearn/A4.png')
plt.show()

# plot classification error
a1 = round(results['validation_0']['error'][2], 4)
a2 = round(results['validation_1']['error'][2], 4)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_axis, results['validation_0']['error'],
        label='Train- %'+str(round(a1*100, 4)))
ax.plot(x_axis, results['validation_1']['error'],
        label='Test-  %'+str(round(a2*100, 4)))
ax.legend(loc="best",fontsize=sizingFont)
plt.title('5-XGBoost Classification Error | '+datasetName, fontsize=sizingFont)
plt.grid(True)
plt.tight_layout()
plt.savefig('../UXviews/A5.png')
plt.savefig('../UXviews/1machLearn/A5.png')
plt.show()
print("==========================|SISFALL_Successfully_Completed!|========================|XGBoost|=================")
