# importiong the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creating the Numpy array
array = np.array([[1, 1, 1], [2, 4, 8], [3, 9, 27],
				[4, 16, 64], [5, 25, 125], [6, 36, 216],
				[7, 49, 343]])

# creating a list of index names
index_values = ['first', 'second', 'third',
				'fourth', 'fifth', 'sixth', 'seventh']

# creating a list of column names
column_values = ['number', 'squares', 'cubes']

# creating the dataframe
df = pd.DataFrame(data = array,
				#index = index_values,
				#columns = column_values
                )

# displaying the dataframe
print(df)
fig, ax1 = plt.subplots(figsize=(14,10))
# fig = plt.figure(figsize=(14, 10))
plt.bar(df,height=70)#, color='gray', height=5,width=0.5, label='Time')
plt.legend(loc='best')
plt.xlabel("Analysing the time computation cost of the different algorithms")
plt.ylabel("Clock time and Accuracy levels")
plt.grid(True)
plt.title("Algorithms and their task completion time period")
plt.savefig('../UXviews/table4/T5d.png')
plt.show()
print('============END========128========')
