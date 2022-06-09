import warnings
import cv2
import matplotlib.pyplot as plt
from pandas.plotting import table
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = [18.50, 5.50]
plt.rcParams["figure.autolayout"] = True
sizingFont = 15
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8]  # Plot-frame
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})  # Inside
print("===================================|ORIGINAL|================================1=============")

df1 = pd.read_csv("ACC/ACCURACYSISFALL.csv", header=1,)
df2 = pd.read_csv("ACC/ACCURACYMOBIACT.csv", header=1,)
df3 = pd.read_csv("ACC/ACCURACYUCIHAR.csv", header=1,)

df1 = df1.iloc[8:9:, -19885:]  # 01-Records, last-TEN headers
df2 = df2.iloc[8:9:, -19885:]  # 01-Records, last-TEN headers
df3 = df3.iloc[8:9:, -19885:]  # 01-Records, last-TEN headers
zed = ['LR Accuracy', 'LDA Accuracy', 'KNN Accuracy', 'DTC Accuracy',
       'ND Accuracy', 'SVM Accuracy', 'RF Accuracy', 'XG Accuracy']
df1.columns = zed
df2.columns = zed
df3.columns = zed
df4 = pd.concat([df1, df2, df3])
lx = ['SISFALL', 'MOBIACT', 'UCIHAR']
cols_swap = lx
df4.insert(0, 'DATASET_used', cols_swap, True)

df5 = df4.T
df4 = pd.DataFrame(df4)
df4.to_csv(r'ACC/ACCURACYALL_Horizontal.csv', index=0)
df5 = pd.DataFrame(df5)
df5.to_csv(r'ACC/ACCURACYALL_Vertical.csv', index=0)

print("===================================|Plotting_FIRST|===========================2=============")

df = pd.DataFrame(df4)
# df.index = ['0001', '0002', '0003']
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'DATASET_X1'}, index={'index': 'NUMBER'})
df.index = df.index + 10001
df.drop('DATASET_X1', axis=1, inplace=True)
print(df)
df.to_csv(r'ACC/ACCURACYALL_Horizontal2.csv', index=0)

fig, ax = plt.subplots(figsize=(16, 3))  # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, df, loc='center', fontsize=25, rowLabels=[
              3]*df.shape[0], colWidths=[0.11]*len(df.columns))  # where df is your data frame
tabla.auto_set_font_size(False)  # Activate set fontsize manually
tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2)  # change size table
plt.grid(True)
plt.suptitle('ACCURACY-HORIZONTAL ALL THREE DATASETS')
plt.title('1-ACCURACY-HORIZONTAL')
plt.tight_layout()
plt.savefig('../UXVIEWS/VH1.png', dpi=99, bbox_inches='tight')
plt.savefig('../UXVIEWS/3ACC/VH1.png', dpi=99, bbox_inches='tight')
plt.show()

print("===================================|Plotting_SECOND|============================3=============")
df = df5
new_header = df.iloc[0]  # grab the first row for the header
df = df[1:]  # take the data less the header row
df = pd.DataFrame(df)
df.columns = new_header  # set the header row as the df header

df.index = ['LR Accuracy', 'LDA Accuracy', 'KNN Accuracy', 'DTC Accuracy',
            'ND Accuracy', 'SVM Accuracy', 'RF Accuracy', 'XG Accuracy']
df.index = ['LogRegression Accuracy', 'LDAnalysis Accuracy', 'KNNeighbour Accuracy', 'DTClassifier Accuracy',
            'GNBayes Accuracy', 'SVMachine Accuracy', 'RandomForest Accuracy', 'XGBoost Accuracy']
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'ALGORITHM USED'}, index={'index': 'NUMBER'})
df.index = df.index + 10001
print(df)
df.to_csv(r'ACC/ACCURACYALL_Vertical2.csv', index=0)

fig, ax = plt.subplots(figsize=(12, 6))  # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
# where df is your data frame
tabla = table(ax, df, loc='center', colWidths=[0.17]*len(df.columns))
tabla.auto_set_font_size(False)  # Activate set fontsize manually
tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2)  # change size table
plt.suptitle('ACCURACY-VERTICAL| All three datasets')
plt.title('2-ACCURACY-VERTICAL| All three datasets')
plt.savefig('../UXVIEWS/3ACC/VH2.png', dpi=99,
            bbox_inches='tight', transparent=True)
plt.show()

print("===================================|Plotting_THIRD|============================4=============")

df2 = df5
df2.index = ['ALGORITHM USED', 'LR Accuracy', 'LDA Accuracy', 'KNN Accuracy',
             'DTC Accuracy', 'ND Accuracy', 'SVM Accuracy', 'RF Accuracy', 'XG Accuracy']
df2.index = ['ALGORITHM USED', 'LogRegression Accuracy', 'LDAnalysis Accuracy', 'KNNeighbour Accuracy',
             'DTClassifier Accuracy', 'GNBayes Accuracy', 'SVMachine Accuracy', 'RandomForest Accuracy', 'XGBoost Accuracy']
new_header = df2.iloc[0]  # grab the first row for the header
df2 = df2[1:]  # take the data less the header row
df2 = pd.DataFrame(df2)
df2.columns = new_header  # set the header row as the df header
df2.reset_index(inplace=True)
df2 = df2.rename(columns={'index': 'ALGORITHM USED'},
                 index={'index': 'NUMBER'})
df2.index = df2.index + 10001
print(df2)
df2.to_csv(r'ACC/ACCURACYALL_Vertical3.csv', index=0)


fig, ax = plt.subplots(figsize=(16, 6))  # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, df2, loc='upper right', colWidths=[
              0.23]*len(df2.columns))  # where df is your data frame
tabla.auto_set_font_size(False)  # Activate set fontsize manually
tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2)  # change size table
plt.suptitle('ACCURACY-SECOND VERTICAL - All three datasets')
plt.title('3-ACCURACY-SECOND VERTICAL')
plt.savefig('../UXVIEWS/3ACC/VH3.png', dpi=99,
            bbox_inches='tight', transparent=True)
plt.show()

print("===================================|s2H_completed_Successufly|=====================|HORIZONTAL|===========")
