# PLOT_DATASET_ONE
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

zed=18
plt.rcParams["figure.figsize"] = [18.50, 5.50]
plt.rcParams["figure.autolayout"] = True
kalas = ['palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green',
         'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green', 'palegreen', 'orange', 'yellow', 'gray', 'skyblue', 'green']

print('===============================|MOBIACT_Both_FallS|==========================================|1|======================')

df11 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FOL/FOL_1_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df12 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FOL/FOL_5_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df13 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FOL/FOL_30_1_annotated.csv', header=0, delimiter=',')  # 30-Years
df14 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FOL/FOL_65_1_annotated.csv', header=0, delimiter=',')  # 40-Years

df15 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FKL/FKL_1_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df16 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FKL/FKL_5_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df17 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FKL/FKL_30_1_annotated.csv', header=0, delimiter=',')  # 30-Years
df18 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/FKL/FKL_65_1_annotated.csv', header=0, delimiter=',')  # 40-Years

df19 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_60_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df20 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_62_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df21 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_63_1_annotated.csv', header=0, delimiter=',')  # 30-Years
df22 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_64_1_annotated.csv', header=0, delimiter=',')  # 40-Years
df23 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_60_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df24 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_62_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df25 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_63_1_annotated.csv', header=0, delimiter=',')  # 30-Years
df26 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_64_1_annotated.csv', header=0, delimiter=',')  # 40-Years

df51 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_40_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df52 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_42_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df53 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_43_1_annotated.csv', header=0, delimiter=',')  # 30-Years
df54 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_44_1_annotated.csv', header=0, delimiter=',')  # 40-Years
df55 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_40_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df56 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_42_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df57 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_43_1_annotated.csv', header=0, delimiter=',')  # 30-Years
df58 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/SDL/SDL_44_1_annotated.csv', header=0, delimiter=',')  # 40-Years

df1 = pd.concat([df11, df12, df13, df14,df15,df16,df17,df18,df19,df20])
df1 = pd.concat([df21,df22,df23,df24,df25,df26,df1,df51,df52,df53,df54,df55,df56,df57,df58])

df21 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STN/STN_1_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df22 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STN/STN_5_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df23 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STN/STN_32_1_annotated.csv', header=0, delimiter=',')  # 30-Years-22222-Missing
df24 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STN/STN_65_1_annotated.csv', header=0, delimiter=',')  # 40-Years
df25 = pd.concat([df21, df22, df23, df24])

df31 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STU/STU_1_1_annotated.csv',header=0, delimiter=',')  # 32-Years
df32 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STU/STU_5_1_annotated.csv', header=0, delimiter=',')  # 36-Years
df33 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STU/STU_32_1_annotated.csv', header=0, delimiter=',')  # 30-Years-22222-Missing
df34 = pd.read_csv('../../MobiAct_Dataset_v2.0/Annotated Data/STU/STU_65_1_annotated.csv', header=0, delimiter=',')  # 40-Years
df2 = pd.concat([df31, df32, df33, df34,df25])
print(df1.shape)
print(df2.shape)


print('===============================|Replace Strings with numbers|================================|2|====================')


df1['label'] = df1['label'].replace({'FKL': 1.0})  # Falling1
df1['label'] = df1['label'].replace({'FOL': 1.0})  # forward-lying3
df1['label'] = df1['label'].replace({'BSC': 1.0})  # LYI2
df1['label'] = df1['label'].replace({'SDL': 1.0})  # LYI2
df1['label'] = df1['label'].replace({'STD': 0.0})  # Standing1
df1['label'] = df1['label'].replace({'LYI': 0.0})  # LYI2
df1['label'] = df1['label'].replace({'STU', 0.0})  # LYI2
df1['label'] = df1['label'].replace({'STN', 0.0})  # LYI2

df2['label'] = df2['label'].replace({'FKL': 1.0})  # Falling1
df2['label'] = df2['label'].replace({'FOL': 1.0})  # Failing2
df2['label'] = df2['label'].replace({'BSC': 1.0})  # LYI2
df2['label'] = df2['label'].replace({'SDL': 1.0})  # LYI2
df2['label'] = df2['label'].replace({'STD': 0.0})  # Standing1
df2['label'] = df2['label'].replace({'LYI': 0.0})  # Standing1
df2['label'] = df2['label'].replace({'STU': 0.0})  # Standing1
df2['label'] = df2['label'].replace({'STN': 0.0})  # Standing1

df2['label'] = df2['label'].astype(int)
df1['label'] = df1['label'].astype(int)
df2['label'] = df2['label'].astype(float)
df1['label'] = df1['label'].astype(float)

print(df1.shape)
print(df2.shape)


print('========================|Copy_Then_Insert_in_1st_position_then_delete_THREE_Field|===============|3|================')
# print((df3))
lx = df1['label']
cols_swap = lx
df1.insert(0, 'Labelz', cols_swap, True)
df1.drop('label', axis=1, inplace=True)
df1.drop('timestamp', axis=1, inplace=True)
df1.drop('rel_time', axis=1, inplace=True)
# print((df3))
lx = df2['label']
cols_swap = lx
df2.insert(0, 'Labelz', cols_swap, True)
df2.drop('label', axis=1, inplace=True)
df2.drop('timestamp', axis=1, inplace=True)
df2.drop('rel_time', axis=1, inplace=True)

print('==================================|make_dataframe_Then_save|====================================[4]==============')
df3 = pd.DataFrame(df1)
df3.to_csv(r'XYZ/XMobiAct1.csv', index=0)
df3 = pd.read_csv('XYZ/XMobiAct1.csv', header=0, delimiter=',')  # 32-Years
df3.to_csv(r'XYZ/XMobiAct2.csv', index=0)
print(df3.shape)

df4 = pd.DataFrame(df2)
df4.to_csv(r'XYZ/XMobiAct3.csv', index=0)
df4 = pd.read_csv('XYZ/XMobiAct3.csv', header=0, delimiter=',')  # 32-Years
df4.to_csv(r'XYZ/XMobiAct4.csv', index=0)
print(df4.shape)

df5 = pd.concat([df3, df4])
df5 = pd.DataFrame(df5)
df5.to_csv(r'XYZ/XMobiAct5.csv', index=0)
df5 = pd.read_csv('XYZ/XMobiAct5.csv', header=0, delimiter=',')  # 32-Years
df5.to_csv(r'XYZ/XMobiAct6.csv', index=0)
print(df5.shape)

df6 = pd.read_csv('XYZ/XMobiAct6.csv', header=1, delimiter=',')  # 32-Years
df6.to_csv(r'XYZ/XMobiAct7.csv', index=0)
print(df6.shape)

print('==================================|Plot_ONE|=================================================[5]============')

df7 = df5.iloc[73:95:, -15:]  # 22-Records, last-15 headers
df8 = df6.iloc[50:72:, -15:]  # 22-Records, last-15 headers
l = len(df7)
c = len(df7.columns)
df7=df7.round(9)
df8=df8.round(9)
df8.columns=df7.columns

# print(df3)
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df7.values, colLabels=df7.columns, loc ='center', colColours=kalas, fontsize=zed)
ax1 = ax.table(cellText=df7.values, colLabels=df7.columns, loc ='center', fontsize=zed)
ax1.auto_set_font_size(False)
ax1.set_fontsize(zed)
plt.title('DATASET_2_MOBIACT | # Headers= '+str(c) + ' / '+str(len(df1.columns)) + ': # Records='+str(l)+'/'+str(len(df1))+'|', fontsize=zed,          
     color='green', fontweight="bold")
plt.tight_layout()
plt.savefig('../UXviews/table1/T8A.png')
plt.show()

print('=======================================|Plot_TWO|========================================[6]=============')
# print(df4)
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax1 = ax.table(cellText=df8.values, colLabels=df8.columns, loc='center', colColours=kalas, fontsize=zed)
ax1 = ax.table(cellText=df8.values, colLabels=df8.columns, loc='center', fontsize=zed)
ax1.auto_set_font_size(False)
ax1.set_fontsize(zed)
plt.title('DATASET_2_MOBIACT | # Headers= '+str(c) + ' / '+str(len(df2.columns)) +': # Records='+str(l)+'/'+str(len(df2))+'|', fontsize=zed, color='green', fontweight="bold")
plt.tight_layout()
plt.savefig('../UXviews/table1/T8B.png')
plt.show()

print(df5.shape)
print(df6.shape)
print(df6.head(5))

print('================================|Statistics_In_Combined_Dataset|==========================[7]=============================================================')
df=df6.loc[df5['Labelz'].isin([1.0])]
index = df.index
number_of_rows1 = len(index)
print('Falls-',number_of_rows1)
print(df.shape)

df7=df6.loc[df5['Labelz'].isin(['0.0'])]
index = df7.index
number_of_rows2 = len(index)
print('ADL-',number_of_rows2)
print(df7.shape)
print('ALL-',number_of_rows1+number_of_rows2)
print(df6.shape)

import inflect
p = inflect.engine()
zed=p.number_to_words(number_of_rows1+number_of_rows2)
print(zed)

print( '=======================|Successfully_complete|===================|pMOBIACT1|==================[8]===========')
