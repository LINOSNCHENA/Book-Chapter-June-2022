import matplotlib.pyplot as plt
import numpy as np

list1=[[0,2],[1,4],[2,5]]
list2=[[0,3.5],[1,9],[2,0.2]]

x1,y1 = zip(*list1)
x2,y2 = zip(*list2)

fig, ax = plt.subplots()
ax.bar(np.array(x1)-0.15, y1, width = 0.3, color='blue')
ax.set_ylabel('List 1', fontsize=16)
ax2 = ax.twinx()
ax2.bar(np.array(x2)+0.15, y2, width = 0.3, color='red')
ax2.set_ylabel('List 2', fontsize=16)
plt.xticks(range(min(x1+x2), max(x1+x2)+1))
plt.show()