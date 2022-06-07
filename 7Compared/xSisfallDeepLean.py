import pandas as pd
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Reshape, SimpleRNN
from sklearn.model_selection import StratifiedKFold
from keras import initializers, losses
from keras.callbacks import ModelCheckpoint, History
from keras.models import load_model
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v2 as tf

# =======================================================================================================
start = time.time()
fontSized = 22
size = 15
seed = 1
batch_size = 1
epochs = 120

np.random.seed(seed)
print("=========================================|SISFALL_DATASET|==========================1==========")
dataset = np.loadtxt('./../5dataXYZ/YSis1ALLs.csv', delimiter=",")
Y = dataset[:, 0:1]
X = dataset[:, 1:10]
print(X.shape)
print(Y.shape)
print(dataset.shape)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
print("=============Kfold=====|", kfold,
      "|============Kfold===========")
for train, test in kfold.split(X, Y):
# model = Sequential()
   model = tf.keras.Sequential()
   LSTM(units=128, input_shape=X.shape[1:])
   model.add(tf.keras.layers.Dense(18)) 
   model.add(tf.keras.layers.Dense(6))
   model.build((Nothers, 9))
   len(model.weights)
   model.compile(optimizer="adam", loss="mse", metrics=["mae","acc","mse"])
   # This builds the model for the first time:
   print(model.metrics_names)
   history = model.fit(X, Y, epochs=epochs, validation_split=0.33, batch_size=batch_size, verbose=1)
   print("=============================================11================================3===============")
   print(history.history.keys())
   print("Training the model.........................................................................")
   history = model.fit(X[train], Y[train], validation_split=0.33,
                        batch_size=batch_size, epochs=epochs, verbose=1)
   scores = model.evaluate(X[test], Y[test], verbose=1)
   print("\n =======Zed metrics = "+"%s: %.2f%%" %
          (model.metrics_names[2], scores[2]*100))
   print("===============ROUNDS==|", epochs,
          "|============ROUNDS====")
   cvscores.append(scores[1] * 100)
   
print("\n ===============================|Parameter|=======================================2===========")
print("%.5f%% (+/- %.5f%%)" % (np.mean(cvscores), np.std(cvscores)))
print(np.mean(cvscores))
print(test)

model.save('Zambia1.h5')
plot_model(model, to_file='../UXViews/others/881A.png',
           show_shapes=True, show_layer_names=True)
del model
print("\n ===========================================|Parameter|==============================3===========")
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

print("\n =======================================|First_Plot|===============================4===========")
# Setting the figure size and Frame
print(accArray1)
print(accArray12)
plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("c")
plt.plot(history.history['acc'], color="blue")
plt.plot(history.history['val_acc'], color="red")
plt.title('81.A1-Sisfall_MobiAct| Model-Accuracy-2.1', fontsize=fontSized)
plt.ylabel('Accuracy values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
plt.legend(['train1,         : %'+str(accArray1),
           'validation1, : %'+str(accArray12)], loc='best', fontsize=size,)
plt.savefig('../UXviews/81A1.png')
plt.savefig('../UXviews/others/81A1.png')
plt.show()


print("\n ===================================|second_Plot|==================================5===========")
print(accArray2)
print(accArray22)
plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("c")
plt.plot(history.history['loss'], color="blue")
plt.plot(history.history['val_loss'], color="red")
plt.title('81.A2-Sisfall_MobiAct | Model-Loss-2.2', fontsize=fontSized)
plt.ylabel('Loss values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
plt.grid(True)
plt.legend(['train2,        :%'+str(accArray2),
           'validation2, : %'+str(accArray22)], loc='best', fontsize=size,)
plt.savefig('../UXviews/81A2.png')
plt.savefig('../UXviews/others/81A2.png')
plt.show()

print("\n ==================================|Third_Plot|====================================6===========")
print(accArray1)
print(accArray32)
plt.figure(figsize=(18, 10))
ax = plt.axes()
ax.set_facecolor("c")
plt.plot(history.history['mse'], color="blue")
plt.plot(history.history['val_mse'], color="fuchsia")
plt.title('81.A3-Sisfall_MobiAct | Model-Error-2.3', fontsize=fontSized)
plt.ylabel('Error values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
plt.grid(True)
# plt.axis([0, epochs, 0.0, 0.5])
plt.legend(['train3,         :%'+str(accArray3),
           'validation3, : %'+str(accArray32)], loc='best', fontsize=size,)
plt.savefig('../UXviews/81A3.png')
plt.savefig('../UXviews/others/81A3.png')
plt.show()


print("\n ================================|Period of Execution|===========================7==========\n")

end = time.time()
print(end - start, '- seconds\n')
print((end - start)/60, '- minutes\n')

print("\n ===============|1_SISFALL_successfully_Completed|====================|others|===========|AAA|==============")
