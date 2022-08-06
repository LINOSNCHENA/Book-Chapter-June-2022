# PLOT_DATASET_Training_1_And_Testing_2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import textwrap

pd.set_option('max_colwidth', 10)
pd.options.display.precision = 15
plt.rcParams['figure.figsize'] = [15, 8]  # Plot-frame
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})  # Inside
fontSizing = 13  # SupTitle & Xlabel
recordWidth = 14
roundOffX = 3
sizingFont=13

kalas = ['palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green']

print("=================================|TABLE1_TRAINING_DATA|========================25==============")
df1 = pd.read_csv("./../6dataXYZ/ucihar4x.csv")
df1 = pd.DataFrame(df1)
A1 = len(df1.loc[df1['ActivityName'].isin(['LAYING'])])
A2 = len(df1.loc[df1['ActivityName'].isin(['STANDING'])])
A3 = len(df1.loc[df1['ActivityName'].isin(['SITTING'])])
A4 = len(df1.loc[df1['ActivityName'].isin(['WALKING'])])
A5 = len(df1.loc[df1['ActivityName'].isin(['WALKING_DOWNSTAIRS'])])
A6 = len(df1.loc[df1['ActivityName'].isin(['WALKING_UPSTAIRS'])])
AZ = [A1, A2, A3, A4, A5, A6]

print('=======================================|Limiting_Number_of_Columns|=================36===============')

df2 = df1.iloc[0:998877128:, -recordWidth:]  # 18-Records, last-TEN headers
df7 = df2.duplicated(subset='ActivityName', keep='last')
df3 = df2.drop_duplicates(subset=["ActivityName"])
df3.drop('subject', inplace=True, axis=1)
df3['Records Per Activity'] = AZ
df3["ActivityName"] = df3["ActivityName"].str.wrap(10)
df3.sort_values(by=['ActivityName'], inplace=True)
df3.sort_values(by=['Records Per Activity'], inplace=True)
df3.style.set_table_styles([dict(selector="th", props=[('max-width', '3px')])])

print('=======================================|TITLE_WRAPPING_ONE|========================49===============')


def wrap(string, max_width):
    return '\n'.join(textwrap.wrap(string, max_width))


currentHeaderX = df3.columns
columnHeaderx_new = []
chomax = 0
for pemba in currentHeaderX:
    s = str(chomax+1)+' - '+currentHeaderX[chomax][-1996:].replace("(vx)", "")
    columnHeaderx_new.append(wrap(s, 10))
    chomax += 1
df3.columns = columnHeaderx_new
df3 = df3.round(roundOffX)
l = len(df3)
c = len(df3.columns)

# fig, ax = plt.subplots(figsize=(18, 2))
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', colColours=kalas, fontsize=sizingFont)
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', fontsize=sizingFont)
ax1.auto_set_font_size(False)
ax1.set_fontsize(fontSizing)
plt.suptitle('../UXviews/tables/TC1.png')
plt.title('1-DATASET_UCIHAR_TRAINING | # Headers= '+str(c) + ' / '+str(len(df1.columns)) +
          ': # Records='+str(l)+'/'+str(len(df1))+'|', fontsize=fontSizing, color='green', fontweight="bold")
plt.tight_layout()
plt.savefig('../UXviews/tables/TC1.png')
plt.show()
print(df3)

print('=======================================|TABLE2_TESTING_DATA|==========================85================')

df1 = pd.read_csv("./../6dataxyz/ucihart2.csv")
A1 = len(df1.loc[df1['ActivityName'].isin(['LAYING'])])
A2 = len(df1.loc[df1['ActivityName'].isin(['STANDING'])])
A3 = len(df1.loc[df1['ActivityName'].isin(['SITTING'])])
A4 = len(df1.loc[df1['ActivityName'].isin(['WALKING'])])
A5 = len(df1.loc[df1['ActivityName'].isin(['WALKING_DOWNSTAIRS'])])
A6 = len(df1.loc[df1['ActivityName'].isin(['WALKING_UPSTAIRS'])])
AZ = [A1, A2, A3, A4, A5, A6]

print('=======================================|Limiting_Number_of_Columns|=================96====================')

df2 = df1.iloc[0:2947:, -recordWidth:]  # 18-Records, last-TEN headers
df7 = df2.duplicated(subset='ActivityName', keep='last')
df3 = df2.drop_duplicates(subset=["ActivityName"])
df3.drop('subject', inplace=True, axis=1)
df3['Records Per Activity'] = AZ
df3.sort_values(by=['Activity'], inplace=True)
df3.sort_values(by=['ActivityName'], inplace=True)
df3.sort_values(by=['Records Per Activity'], inplace=True)
df3["ActivityName"] = df3["ActivityName"].str.wrap(10)

print('=======================================|TITLE_WRAPPING_TWO|========================108=====================')


def wrap(string, max_width):
    return '\n'.join(textwrap.wrap(string, max_width))


currentHeaderX = df3.columns
columnHeaderx_new = []
chomax = 0
for pemba in currentHeaderX:
    s = str(chomax+1)+' - '+currentHeaderX[chomax][-1996:].replace("(vx)", "")
    columnHeaderx_new.append(wrap(s, 10))
    chomax += 1
df3.columns = columnHeaderx_new
df3 = df3.round(roundOffX)
l = len(df3)
c = len(df3.columns)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', colColours=kalas, fontsize=sizingFont)
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', fontsize=sizingFont)
ax1.auto_set_font_size(False)
ax1.set_fontsize(fontSizing)
plt.suptitle('../UXviews/tables/TC2.png')
plt.title('2-DATASET_UCIHAR_TESTING | # Headers= '+str(c) + ' / '+str(len(df1.columns)) +
          ': # Records='+str(l)+'/'+str(len(df1))+'|', fontsize=fontSizing, color='green', fontweight="bold")
plt.tight_layout()
plt.savefig('../UXviews/tables/TC2.png')
plt.show()

print("===================================|UCIHAR_Plotting_completed_Successufly|==============143=============")
print(df1.shape)
print(df2.shape)
print(df3.shape)

print("===================================|UCIHAR_Plotting_completed_Successufly|==============148==============")
