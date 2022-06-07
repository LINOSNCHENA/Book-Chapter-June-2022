# PLOT_DATASET_ONE
import textwrap
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pd.set_option('max_colwidth', 10)
pd.options.display.precision = 3
plt.rcParams['figure.figsize'] = [15, 8]  # Plot-frame
# plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 29})  # Inside
fondSizing = 15  # SupTitle & Xlabel
recordsHieght = 14

print("=================================|Select_DataFrame_Source|========================1==============")

df1 = pd.read_csv("./../6dataXYZ/ucihar4x.csv")
df1 = pd.DataFrame(df1)
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

print('=======================================|Activity_Counter|==========================2==============')
A1 = len(df1.loc[df1['ActivityName'].isin(['LAYING'])])
A2 = len(df1.loc[df1['ActivityName'].isin(['STANDING'])])
A3 = len(df1.loc[df1['ActivityName'].isin(['SITTING'])])
A4 = len(df1.loc[df1['ActivityName'].isin(['WALKING'])])
A5 = len(df1.loc[df1['ActivityName'].isin(['WALKING_DOWNSTAIRS'])])
A6 = len(df1.loc[df1['ActivityName'].isin(['WALKING_UPSTAIRS'])])
AZ = [A1, A2, A3, A4, A5, A6]
print('=======================================|Limiting_Number_of_Columns|=================3===============')
df2 = df1.iloc[0:998877128:, -recordsHieght:]  # 18-Records, last-TEN headers
df7 = df2.duplicated(subset='ActivityName', keep='last')
df3 = df2.drop_duplicates(subset=["ActivityName"])
df3.drop('subject', inplace=True, axis=1)
df3['Records Per Activity'] = AZ
df3["ActivityName"] = df3["ActivityName"].str.wrap(10)
print('=======================================|TITLE_Number_TITLE|========================4===============')
# df3.sort_values(by=['Activity'], inplace=True)
df3.sort_values(by=['ActivityName'], inplace=True)
df3.sort_values(by=['Records Per Activity'], inplace=True)
df3.style.set_table_styles([dict(selector="th", props=[('max-width', '3px')])])

print('=======================================|TITLE_WRAPPING_ONE|========================5===============')

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
df3 = df3.round(3)
l = len(df3)
c = len(df3.columns)
print(df1.shape)
print(df2.shape)
print(df3.shape)

fig, ax = plt.subplots(figsize=(18, 2))
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', colColours=kalas, fontsize=15)
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', fontsize=15)
ax1.auto_set_font_size(False)
ax1.set_fontsize(10)
plt.title('DATASET_UCIHAR_TRAINING | # Headers= '+str(c) + ' / '+str(len(df1.columns)) +
          ': # Records='+str(l)+'/'+str(len(df1))+'|', fontsize=fondSizing, color='green', fontweight="bold")
plt.tight_layout()
fig.tight_layout()
plt.savefig('../UXviews/tables/TC1.png')
plt.show()

print('=======================================|Table_TESTING_DATA|==========================7T1================')

df1 = pd.read_csv("./../6dataxyz/ucihart2.csv")
A1 = len(df1.loc[df1['ActivityName'].isin(['LAYING'])])
A2 = len(df1.loc[df1['ActivityName'].isin(['STANDING'])])
A3 = len(df1.loc[df1['ActivityName'].isin(['SITTING'])])
A4 = len(df1.loc[df1['ActivityName'].isin(['WALKING'])])
A5 = len(df1.loc[df1['ActivityName'].isin(['WALKING_DOWNSTAIRS'])])
A6 = len(df1.loc[df1['ActivityName'].isin(['WALKING_UPSTAIRS'])])
AZ = [A1, A2, A3, A4, A5, A6]
print(sum(AZ))
print(df1.shape)

print('=======================================|Limiting_Number_of_Columns|=================8T2===============')

df2 = df1.iloc[0:2947:, -recordsHieght:]  # 18-Records, last-TEN headers
df7 = df2.duplicated(subset='ActivityName', keep='last')
df3 = df2.drop_duplicates(subset=["ActivityName"])
df3.drop('subject', inplace=True, axis=1)
df3['Records Per Activity'] = AZ
df3.sort_values(by=['Activity'], inplace=True)
df3.sort_values(by=['ActivityName'], inplace=True)
df3.sort_values(by=['Records Per Activity'], inplace=True)
df3["ActivityName"] = df3["ActivityName"].str.wrap(10)

print('=======================================|TITLE_WRAPPING_TWO|========================9T3===============')

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
df3 = df3.round(9)
l = len(df3)
c = len(df3.columns)

fig, ax = plt.subplots(figsize=(18, 2))
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', colColours=kalas, fontsize=15)
ax1 = ax.table(cellText=df3.values, colLabels=df3.columns,
               loc='center', fontsize=15)
ax1.auto_set_font_size(False)
ax1.set_fontsize(12)
plt.title('DATASET_UCIHAR_TESTING | # Headers= '+str(c) + ' / '+str(len(df1.columns)) +
          ': # Records='+str(l)+'/'+str(len(df1))+'|', fontsize=fondSizing, color='green', fontweight="bold")

plt.tight_layout()
fig.tight_layout()
plt.savefig('../UXviews/tables/TC2.png')
plt.show()

print("===================================|UCIHAR_Plotting_completed_Successufly|==============11===========")
