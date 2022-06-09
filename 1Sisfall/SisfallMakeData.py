# PLOT_DATASET_ONE
import inflect
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
sizingFont = 15

print('======================|Headings_six_Records_20|====================[1]===============')

plt.rcParams["figure.figsize"] = [18.50, 5.50]
plt.rcParams["figure.autolayout"] = True
kalas = ['palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green']

print('=================================|SISFALL_FALL|===========================|2|=============')
names1 = ["x_acc", "y_acc", "z_acc", "x_gyr",
          "y_grc", "z_grc", 'Azimuth', 'Pitch', 'Roll']
df10 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/F01_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")
df11 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/F02_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")
df12 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/F11_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")
df13 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/F14_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")

ONE1 = [1] * len(df10)  # FALL -forward slip
ONE2 = [1] * len(df11)  # FALL -backwardslip
ONE3 = [1] * len(df12)  # FALL -Backward trying sit
ONE4 = [1] * len(df13)  # FALL -Backward when sitting
df10['Sisfall_Label_X4'] = ONE1
df11['Sisfall_Label_X4'] = ONE2
df12['Sisfall_Label_X4'] = ONE3
df13['Sisfall_Label_X4'] = ONE4
df4 = pd.concat([df10, df11, df12, df13])
lx = df4['Sisfall_Label_X4']
cols_swap = lx
df4.insert(0, 'labelz', cols_swap, True)
df4.drop('Sisfall_Label_X4', axis=1, inplace=True)

df4 = pd.DataFrame(df4)
df4.to_csv(r'./../6dataXYZ/YSis1falls.csv', index=0)

print('=========================================|SISFALL_ADL|==============================|3|=============')
df10 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/D01_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")
df11 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/D02_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")
df12 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/D11_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")
df13 = pd.read_table('../../MobiAct_Dataset_v2.0/SisFall_dataset/SE06/D14_SE06_R01.txt',
                     header=1, names=names1, sep=',', escapechar=";")

ONE1 = [0] * len(df10)  # ADL -Walking fast
ONE2 = [0] * len(df11)  # ADL -walking quick
ONE3 = [0] * len(df12)  # ADL -Sitting
ONE4 = [0] * len(df13)  # ADL -Turning
df10['Sisfall_Label_X4'] = ONE1
df11['Sisfall_Label_X4'] = ONE2
df12['Sisfall_Label_X4'] = ONE3
df13['Sisfall_Label_X4'] = ONE4
df4 = pd.concat([df10, df11, df12, df13])
lx = df4['Sisfall_Label_X4']
cols_swap = lx
df4.insert(0, 'labelz', cols_swap, True)
df4.drop('Sisfall_Label_X4', axis=1, inplace=True)  # Remove Colum x4
df4.drop(df4.loc[df4['labelz'] == 'labelz'].index,
         inplace=True)  # Remove Row z
df4 = pd.DataFrame(df4)
df4.to_csv(r'./../6dataXYZ/YSis1ADLs.csv', index=0)


print(
    '====================================|COMBINED_SISFALL|====================================[4]=============')
df31 = pd.read_csv('./../6dataXYZ/YSis1Falls.csv', header=0, delimiter=',')
df32 = pd.read_csv('./../6dataXYZ/YSis1ADLs.csv', header=0, delimiter=',')
df33 = pd.concat([df31, df32])
df34 = df33
df2 = df33.iloc[0:23:, -15:]  # 22-Records, last-12 headers
df2 = df2.round(9)

l = len(df2)
c = len(df2.columns)
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df2.values, colLabels=df2.columns,
               loc='center', colColours=kalas, fontsize=sizingFont)
ax1 = ax.table(cellText=df2.values, colLabels=df2.columns,
               loc='center', fontsize=sizingFont)
ax1.auto_set_font_size(False)
ax1.set_fontsize(sizingFont)
plt.suptitle('../UXviews/tables/TA1.png')
plt.title('DATASET_1_SISFALL | # Headers= '+str(c) + ' / '+str(len(df33.columns)) +
          ': # Records='+str(l)+'/'+str(len(df33))+'|', fontsize=sizingFont, color='green', fontweight="bold")
plt.tight_layout()
plt.savefig('../UXviews/tables/TA1.png')
plt.show()

df2 = pd.DataFrame(df34)
print(df2.shape)
df2.to_csv(r'./../6dataXYZ/YSis1ALLS.csv', index=0)

print(
    '===========================================|Plot_Combined|=======================[5]=============')
df4 = df2.iloc[11980:12001:, -15:]  # 22-Records, last-15X headers
l = len(df4)
c = len(df4.columns)
df4 = df4.round(9)
# print(df2)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df4.values, colLabels=df4.columns,
               loc='center', colColours=kalas, fontsize=sizingFont)
ax1 = ax.table(cellText=df4.values, colLabels=df4.columns,
               loc='center', fontsize=sizingFont)
ax1.auto_set_font_size(False)
ax1.set_fontsize(sizingFont)
plt.suptitle('../UXviews/tables/TA2.png')
plt.title('DATASET_1_SISFALL | # Headers= '+str(c) + ' / '+str(len(df2.columns)) +
          ': # Records='+str(l)+'/'+str(len(df2))+'|', fontsize=sizingFont, color='green', fontweight="bold")
plt.tight_layout()
plt.savefig('../UXviews/tables/TA2.png')
plt.show()
print('=======================================|SISFALL_THREE_SMALL|===========================|6|=============')
df10 = pd.read_csv('./../6dataXYZ/YSis1ALLS.csv', header=0, delimiter=',')
print(df10)
print(df10.columns)
df11 = df10
df4 = df10.iloc[0:25000:, -15:]  # 22-Records, last-15X headers
df4.loc[df4['labelz'].isin([0])]  # If contains one add to df4
df4.to_csv(r'./../6dataXYZ/YSisFall21BZ.csv', header=0, index=0)

df6 = df10.iloc[25000:40000:, -15:]  # 22-Records, last-15X headers
df6.loc[df6['labelz'].isin([1])]
df6.to_csv(r'./../6dataXYZ/YSisFall22BP.csv', header=0, index=0)

df7 = df10.iloc[40000:50000:, -15:]  # 22-Records, last-15X headers
df7.loc[df7['labelz'].isin([2])]
df7.to_csv(r'./../6dataXYZ/YSisFall23B2.csv', header=0, index=0)

print('=========================================|SISFALL_THREE_BIG|==========================|7|=============')

df4 = df10.iloc[0:35000:, -15:]  # 22-Records, last-15X headers
df4 = df4.loc[df4['labelz'].isin([1])]
df4 = df4.iloc[0:55:, -15:]      # 22-Records, last-15X headers
df4.to_csv(r'./../6dataXYZ/YSisFall31S1.csv', header=0, index=0)

df6 = df10.iloc[11982:45010:, -15:]  # 22-Records, last-15X headers
df6 = df6.iloc[0:55:, -15:]      # 22-Records, last-15X headers
df6.to_csv(r'./../6dataXYZ/YSisFall32SX.csv', header=0, index=0)

df7 = df10.iloc[0:50000:, -15:]   # 22-Records, last-15X headers
df7 = df7.loc[df7['labelz'].isin([0])]
df7 = df7.iloc[0:55:, -15:]       # 22-Records, last-15X headers
df7.to_csv(r'./../6dataXYZ/YSisFall33SZ.csv', header=0, index=0)


print('=================================|SISFALL_REMOVE_HEADERS|===============================|8|=============')

df10.to_csv(r'./../6dataXYZ/YSis1ALLS.csv', header=0, index=0)
print(df7.head(5))

print(
    '==============================|Statistics_In_ALL_Dataset|===============================[9]=============')

print(df11.head(5))
df6 = df11.loc[df11['labelz'].isin([1])]
index = df6.index
number_of_rows1 = len(index)
print('Falls-', number_of_rows1)
print(df6.shape)

df7 = df11.loc[df11['labelz'].isin([0])]
index = df7.index
number_of_rows2 = len(index)
print('ADL-', number_of_rows2)
print(df7.shape)
print('ALL-', number_of_rows1+number_of_rows2)
print(df10.shape)

p = inflect.engine()
sizingFont = p.number_to_words(number_of_rows1+number_of_rows2)
print(sizingFont)

print(
    '=========================|Successfuly_Completed|===============|1SISFALL1|============[10]==============')
