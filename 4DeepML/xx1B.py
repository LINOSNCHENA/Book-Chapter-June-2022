import numpy
import numpy as np
import time
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import shap
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
fontSized = 22
size = 15
start = time.time()
n_features = 10
seed = 1
numpy.random.seed(seed)
batch_size = 1
epochs = 150
plotName="Deep-Learning-MobiAct-dataset "

model = models.Sequential(name="DeepNN", layers=[
    # hidden layer 1
    layers.Dense(name="h1", input_dim=n_features,
                 units=int(round((n_features+1)/2)),
                 activation='relu'),
    layers.Dropout(name="drop1", rate=0.2),

    # hidden layer 2
    layers.Dense(name="h2", units=int(round((n_features+1)/4)),
                 activation='relu'),
    layers.Dropout(name="drop2", rate=0.2),

    # layer output
    layers.Dense(name="output", units=1, activation='sigmoid')
])
model.summary()

# Perceptron
inputs = layers.Input(name="input", shape=(3,))
outputs = layers.Dense(name="output", units=1,
                       activation='linear')(inputs)
model = models.Model(inputs=inputs, outputs=outputs,
                     name="Perceptron")

# DeepNN
# layer input
inputs = layers.Input(name="input", shape=(n_features,))
# hidden layer 1
h1 = layers.Dense(name="h1", units=int(
    round((n_features+1)/2)), activation='relu')(inputs)
h1 = layers.Dropout(name="drop1", rate=0.2)(h1)
# hidden layer 2
h2 = layers.Dense(name="h2", units=int(
    round((n_features+1)/4)), activation='relu')(h1)
h2 = layers.Dropout(name="drop2", rate=0.2)(h2)
# layer output
outputs = layers.Dense(name="output", units=1, activation='sigmoid')(h2)
model = models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
# fix random seed for reproducibility
print("=========================================|MOBIACT_DATASET|=======================1==============")
dataset = np.loadtxt('./../5dataXYZ/XMobiAct7.csv', delimiter=",")
Y = dataset[:, 0:1]
X = dataset[:, 0:10]
print(X.shape)
print(Y.shape)
print(dataset.shape)

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
plot_model(model, to_file='../UXViews/image4/21A.png',
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
plt.title('11.A1-Sequential2021_Data | Model-Accuracy-2.1', fontsize=fontSized)
plt.title(str(plotName)+' | Model-Accuracy', fontsize=fontSized)
plt.ylabel('Accuracy values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
# plt.axis([0, epochs, 0.0, 1.1])
plt.legend(['train1,         : %'+str(accArray1),
           'validation1, : %'+str(accArray12)], loc='best', fontsize=size,)
plt.savefig('../UXviews/82A1.png')
plt.savefig('../UXviews/image4/21b.png')
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
plt.title('12.B2-Sequential2021_Data | Model-Loss-2.2', fontsize=fontSized)
plt.title(str(plotName)+' | Model-Loss', fontsize=fontSized)
plt.ylabel('Loss values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
plt.grid(True)
# plt.axis([0, epochs, 0, 0.99])
plt.legend(['train2,        :%'+str(accArray2),
           'validation2, : %'+str(accArray22)], loc='best', fontsize=size,)
plt.savefig('../UXviews/82A2.png')
plt.savefig('../UXviews/image4/21c.png')
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
plt.title('13-Sequential2021_Data | Model-Error-2.3', fontsize=fontSized)
plt.title(str(plotName)+' | Model-Error', fontsize=fontSized)
plt.ylabel('Error values', fontsize=fontSized)
plt.xlabel('epoch', fontsize=fontSized)
# plt.axis([0, epochs, 0, 0.99])
plt.legend(['train3,         :%'+str(accArray3),
           'validation3, : %'+str(accArray32)], loc='best', fontsize=size,)
plt.savefig('../UXviews/82A3.png')
plt.savefig('../UXviews/image4/21d.png')
plt.show()

print("\n ===============================|Period of Execution|====================10============\n")

end = time.time()
print(end - start, '- seconds\n')
print((end - start)/60, '- minutes\n')


print("\n ===============|2_MOBIACT_Successfully_Completed|===================|secondXX1|==============|BBB|==========")
