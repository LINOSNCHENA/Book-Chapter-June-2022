import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.utils.vis_utils import plot_model
import numpy as np
import time
from keras.models import model_from_json
import matplotlib.pyplot as plt


start = time.time()
fontSized = 22
size = 15
seed = 1
batch_size = 1
epochs = 8


dataset = np.loadtxt('./XYZ/YSis1ALLs.csv', delimiter=",")
Y = dataset[:, 0:1]
X = dataset[:, 1:10]


# When using the delayed-build pattern (no input shape specified), you can
# choose to manually build your model by calling
# `build(batch_input_shape)`:
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(18))
model.add(tf.keras.layers.Dense(6))
model.build((None, 9))
len(model.weights)
# Returns "4"

# Note that when using the delayed-build pattern (no input shape specified),
# the model gets built the first time you call `fit`, `eval`, or `predict`,
# or the first time you call the model on some input data.
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(8))
# model.add(tf.keras.layers.Dense(1))
print("=============================================11================================1===============")
#model.compile(optimizer='sgd', loss='mse',metrics=["mae", "acc","mse"])
model.compile(optimizer="adam", loss="mse", metrics=["mae","acc","mse"])
# This builds the model for the first time:
print(model.metrics_names)
# model.fit(X, Y, batch_size=32, epochs=2)


print("=============================================11================================2===============")
history = model.fit(X, Y, epochs=epochs,
                    validation_split=0.33, batch_size=batch_size, verbose=1)
print(history.history.keys())
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = model.to_json()
with open("XmodelOne.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("XmodelOne.h5")
print("Saved model to disk")

plot_model(model, to_file='../UXViews/EIGHT/882C.png',    show_shapes=True, show_layer_names=True)
print("=============================================11================================3===============")

print(model.metrics_names)
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
scores = model.evaluate(X, Y, verbose=0)

print("=============================================11================================4===============")
allSizes = (history.history['acc'])
arrLengthX = len(allSizes)

accArray1 = (history.history['acc'])
accArray1 = round(100*accArray1[arrLengthX-1], 4)
accArray12 = (history.history['val_acc'])
accArray12 = round(100*accArray12[arrLengthX-1], 4)


accArray2 = (history.history['mse'])
accArray2 = round(100*accArray2[arrLengthX-1], 4)
accArray22 = (history.history['val_mse'])
accArray22 = round(100*accArray22[arrLengthX-1], 4)


accArray3 = (history.history['loss'])
accArray3 = round(100*accArray3[arrLengthX-1], 4)
accArray32 = (history.history['val_loss'])
accArray32 = round(100*accArray32[arrLengthX-1], 4)

print("\n ===============|3|======================|First_Plot|============================5===============")
print(history.history.keys())
# Setting the figure size and Frame
print(accArray1)
print(accArray12)

plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("beige")
plt.plot(history.history['acc'], color="blue")
plt.plot(history.history['val_acc'], color="red")
plt.title('82.C1-Sequential2021_Data | Model-Accuracy-2.1', fontsize=fontSized)
plt.ylabel('Accuracy values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
# plt.axis([0, epochs, 0.0, 1.1])
plt.legend(['train1,         : %'+str(accArray1),
           'validation1, : %'+str(accArray12)], loc='best', fontsize=size,)
plt.savefig('../UXviews/8A1.png')
plt.savefig('../UXviews/EIGHT/8A1.png')
plt.show()


print("\n ===============|4|======================|second_Plot|===========================6==============")
# Setting the figure size and Frame
print(accArray2)
print(accArray22)


plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("beige")
plt.plot(history.history['loss'], color="fuchsia")
plt.plot(history.history['val_loss'], color="blue")
plt.title('82.C2-Sequential2021_Data | Model-Loss-2.2', fontsize=fontSized)
plt.ylabel('Loss values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
plt.grid(True)
# plt.axis([0, epochs, 0, 0.99])
plt.legend(['train2,        :%'+str(accArray2),
           'validation2, : %'+str(accArray22)], loc='best', fontsize=size,)
plt.savefig('../UXviews/8A2.png')
plt.savefig('../UXviews/EIGHT/8A2.png')
plt.show()


print("\n ===============|5|====================|Third_Plot|=================================7=============")
# Setting the figure size and Frame
print(accArray3)
print(accArray32)

plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("beige")
plt.plot(history.history['mse'], color="blue")
plt.plot(history.history['val_mse'], color="fuchsia")
plt.title('82.C3-Sequential2021_Data | Model-Error-2.3', fontsize=fontSized)
plt.ylabel('Error values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
# plt.axis([0, epochs, 0, 0.99])
plt.legend(['train3,         :%'+str(accArray3),
           'validation3, : %'+str(accArray32)], loc='best', fontsize=size,)
plt.savefig('../UXviews/8A3.png')
plt.savefig('../UXviews/EIGHT/8A3.png')
plt.show()

print("\n ===============================|Period of Execution|===============================8============\n")

end = time.time()

print(end - start, '- seconds\n')
print((end - start)/60, '- minutes\n')

print("\n ===============|2_SISFALL_ibeacon_Successfully_Completed|===================|ZAMB_ONE|==============|AAA|==========")
