import pandas as pd
data = pd.read_csv('diabetes.csv')
x = data.drop("Outcome", axis=1)
y = data["Outcome"]
print(y)
print(data)

# from keras.models import Sequential
# from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable warning for New Tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x,y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(x, y)
print("Model accuracy: %.2f"% (accuracy*100))
predictions = model.predict(x)
print([round(x[0]) for x in predictions])