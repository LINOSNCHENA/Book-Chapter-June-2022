import pandas as pd
import numpy as np
import os

# get the features from the file features.txt
print('=========================|1_Count_size_Of_Lines|===================================================1=============')
features = list()
with open('UCIHAR/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]
print('Number of Column headers in UCIHAR dataset-:', len(features))

print('=========================|2_Count_Number_Records|==================================================2=============')

# get the data from txt files to pandas dataffame
X_train = pd.read_csv('UCIHAR/train/X_train.txt',
                      delim_whitespace=True, header=None)
print('Number of Rows for Training in UCIHAR dataset-:', len(X_train))
X_train.columns = features
X_train.head(2)
print(X_train.head(2))
print(X_train.shape)

# add subject column to the dataframe
print('=========================|3_Add_Label_As_Subject_sorted|==========================================3============')
X_train['subject'] = pd.read_csv(
    'UCIHAR/train/subject_train.txt', header=None, squeeze=True)
print(X_train.head(2))
y_train = pd.read_csv('UCIHAR/train/y_train.txt',
                      names=['Activity'], squeeze=True)
print(np.sort(y_train.unique()))
y_train_labels = y_train.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS',
                             3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
print(np.sort(y_train_labels.unique()))

print('=========================|4_Combined_Train_Activity_NameOfActivity|==============================4============')
# put all columns in a single dataframe
train = X_train
train = train.iloc[0:99917999:, -9110:]

train['Activity'] = y_train
train['ActivityName'] = y_train_labels
train.sample()
train.shape
print(train.head(5))

print('================================|5_Add_Same_Headings_To_Test_Data|===============================5===========')
# get the data from txt files to pandas dataffame
X_test = pd.read_csv('UCIHAR/test/X_test.txt',
                     delim_whitespace=True, header=None)
X_test.columns = features
X_test.head()
print(X_test.head(5))

print('================================|6_Repeat_Process_Test_Like_Training_Dataset|====================6===========')
# add subject column to the dataframe
X_test['subject'] = pd.read_csv(
    'UCIHAR/test/subject_test.txt', header=None, squeeze=True)
y_test = pd.read_csv('UCIHAR/test/y_test.txt',
                     names=['Activity'], squeeze=True)
y_test_labels = y_test.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS',
                           3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
# put all columns in a single dataframe
test = X_test
test = test.iloc[0:99917999:, -9110:]
test['Activity'] = y_test
test['ActivityName'] = y_test_labels
test.sample()

print('Number of duplicates in train -: {}'.format(sum(train.duplicated())))
print('Nnumber of duplicates in test -: {}'.format(sum(test.duplicated())))
print('Number of NaN/Null values in train -: {}'.format(train.isnull().values.sum()))
print('Number of NaN/Null values in test  -: {}'.format(test.isnull().values.sum()))

print('==================================|7_FINAL_SIZES_AND_RESULTS|==================================7============')
print(train.shape)
print(test.shape)

print('==================================|8_FINAL_SIZES_AND_RESULTS|==================================8============')
zed = train['Activity']
print(zed.unique())

df = train.loc[train['Activity'].isin([6.0])]
print(df.head(2))
print(df.shape)
train['Activity'] = train['Activity'].replace({1.0: 0.0})  # Falling1
train['Activity'] = train['Activity'].replace({2.0: 0.0})  # Standing1
train['Activity'] = train['Activity'].replace({3.0: 0.0})  # LYI2
train['Activity'] = train['Activity'].replace({4.0: 0.0})  # LYI2
train['Activity'] = train['Activity'].replace({5.0: 0.0})  # LYI2
train['Activity'] = train['Activity'].replace({6.0: 1.0})  # LYI2


test['Activity'] = test['Activity'].replace({1.0: 0.0})  # Falling1
test['Activity'] = test['Activity'].replace({2.0: 0.0})  # Standing1
test['Activity'] = test['Activity'].replace({3.0: 0.0})  # Standing1
test['Activity'] = test['Activity'].replace({4.0: 0.0})  # Standing1
test['Activity'] = test['Activity'].replace({5.0: 0.0})  # Standing1
test['Activity'] = test['Activity'].replace({6.0: 1.0})  # Standing1

zed = train['Activity']
print(zed.unique())

print('==============================|9_REPLACEMENT_ACTIVITY_ONE_AND_ZEROS_RESULTS|================9===============')
train.drop('Activity', axis=1, inplace=True)
train.insert(0, 'Activity', zed, True)

print(train.shape)
print(test.shape)

print('==================================|10_ONES_VS_ZEROS_RESULTS|==================================10===============')
df = train.loc[train['Activity'].isin([1.0])]
print(df.shape)
train.to_csv('ucihar1.csv', header=1, index=0)  # TRAIN-2
test.to_csv('ucihart2.csv', header=1, index=0)    # TEST-2
train.drop('ActivityName', axis=1, inplace=True)
train.drop('subject', axis=1, inplace=True)

train.to_csv('./../5DataXYZ/ucihar1.csv', header=1, index=0)
test.to_csv('./../5dataXYZ/ucihart2.csv', header=1, index=0)
print(train.head(4))

print('===============================|11_REMOVE_WORDS_AND_HEADERS|==================================10===============')

df = train.loc[train['Activity'].isin([0.0])]
print(df.shape)
print(train.shape)
print(train)
train.to_csv('./../5DataXYZ/ucihar1.csv', header=1, index=0)
test.to_csv('./../5DataXYZ/ucihar2.csv', header=1, index=0)

print('==================================|First_Successfully_Completed|===============================11===============')
