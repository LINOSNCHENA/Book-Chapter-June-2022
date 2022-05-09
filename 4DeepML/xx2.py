import os
import time
import numpy
import numpy as np
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow.compat.v2 as tf
import keras
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
start = time.time()
fontSized = 22
size = 15
seed = 1
batch_size = 1
epochs = 31

# fix random seed for reproducibility
numpy.random.seed(seed)
print("=========================================|MOBIACT_DATASET|=======================1==============")
dataset = np.loadtxt('./../5dataXYZ/YSis1ALLs.csv', delimiter=",")
Y = dataset[:, 0:1]
X = dataset[:, 1:10]
print(X.shape)
print(Y.shape)
print(dataset.shape)

# create model
print("==============================================11================================2===============")
# model = Sequential()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(18))
model.add(tf.keras.layers.Dense(6))
model.build((None, 9))
len(model.weights)


model.compile(optimizer="adam", loss="mean_squared_error",
              metrics=["mae", "acc", "mean_squared_error"])
# This builds the model for the first time:
print(model.metrics_names)
history = model.fit(X, Y, epochs=epochs, validation_split=0.33,
                    batch_size=batch_size, verbose=1)
print("=============================================11================================3===============")
print(history.history.keys())
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# serialize model to JSON
plot_model(model, to_file='../UXViews/image4/2A.png',
           show_shapes=True, show_layer_names=True)
print("=============================================11================================4===============")
print(model.metrics_names)
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
scores = model.evaluate(X, Y, verbose=0)
print("=============================================11================================6===============")
allSizes = (history.history['acc'])
arrLengthX = len(allSizes)

accArray1 = (history.history['acc'])
accArray1 = round(100*accArray1[arrLengthX-1], 4)
accArray12 = (history.history['val_acc'])
accArray12 = round(100*accArray12[arrLengthX-1], 4)


accArray2 = (history.history['mean_squared_error'])
accArray2 = round(100*accArray2[arrLengthX-1], 4)
accArray22 = (history.history['val_mean_squared_error'])
accArray22 = round(100*accArray22[arrLengthX-1], 4)


accArray3 = (history.history['loss'])
accArray3 = round(100*accArray3[arrLengthX-1], 4)
accArray32 = (history.history['val_loss'])
accArray32 = round(100*accArray32[arrLengthX-1], 4)

print("\n ===============|3|======================|First_Plot|=====================7===============")
print(history.history.keys())
print(accArray1)
print(accArray12)

# Setting the figure size and Frame
plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("beige")
plt.plot(history.history['acc'], color="blue")
plt.plot(history.history['val_acc'], color="red")
plt.title('22A1-Sequential2021_Data | Model-Accuracy-2.1', fontsize=fontSized)
plt.ylabel('Accuracy values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
# plt.axis([0, epochs, 0.0, 1.1])
plt.legend(['train1,         : %'+str(accArray1),
           'validation1, : %'+str(accArray12)], loc='best', fontsize=size,)
plt.savefig('../UXviews/82A1.png')
plt.savefig('../UXviews/image4/2b.png')
plt.show()


print("\n ===============|4|======================|second_Plot|===================8==============")
# Setting the figure size and Frame
print(accArray2)
print(accArray22)

plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("beige")
plt.plot(history.history['loss'], color="fuchsia")
plt.plot(history.history['val_loss'], color="blue")
plt.title('22.B2-Sequential2021_Data | Model-Loss-2.2', fontsize=fontSized)
plt.ylabel('Loss values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
plt.grid(True)
# plt.axis([0, epochs, 0, 0.99])
plt.legend(['train2,        :%'+str(accArray2),
           'validation2, : %'+str(accArray22)], loc='best', fontsize=size,)
plt.savefig('../UXviews/82A2.png')
plt.savefig('../UXviews/image4/2c.png')
plt.show()


print("\n ===============|5|====================|Third_Plot|=========================9=============")
# Setting the figure size and Frame
print(accArray3)
print(accArray32)

plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("beige")
plt.plot(history.history['mean_squared_error'], color="blue")
plt.plot(history.history['val_mean_squared_error'], color="fuchsia")
plt.title('22.A3-Sequential2021_Data | Model-Error-2.3', fontsize=fontSized)
plt.ylabel('Error values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
# plt.axis([0, epochs, 0, 0.99])
plt.legend(['train3,         :%'+str(accArray3),
           'validation3, : %'+str(accArray32)], loc='best', fontsize=size,)
plt.savefig('../UXviews/82A3.png')
plt.savefig('../UXviews/image4/2d.png')
plt.show()

print("\n ===============================|Period of Execution|====================10============\n")

end = time.time()
print(end - start, '- seconds\n')
print((end - start)/60, '- minutes\n')


print("\n ===============|2_MOBIACT_SECOND_Successfully_Completed|===================|sECOND-XX2|==============|AAA|==========")
