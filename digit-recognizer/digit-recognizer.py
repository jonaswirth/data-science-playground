import keras
import matplotlib.pyplot as plt
from keras import Sequential
import numpy as np
from keras.layers import Input, Activation, Dense, Dropout
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('digit-recognizer/data/train.csv')
X_test = pd.read_csv('digit-recognizer/data/test.csv')
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

# Normalize to floats in [0:1] --> Why does this significantly improve performance?
X_train = X_train.astype('float32') / 255

# Hot encode labels
y_train = np.array(y_train).reshape(-1, 1)
y_train = OneHotEncoder(sparse=False).fit_transform(y_train)

# Show image
# img = np.array(X_train.iloc[42,:])
# img = img.reshape((-1,28))
# plt.imshow(img)
# plt.show()
# print(y_train[42])

model = Sequential()
model.add(Input(shape=(784)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(rate=0.2))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=15, batch_size=20, validation_split=.15)

model.save("digit-recognizer/model.keras")

y_pred = model.predict(X_test)

y_pred = np.array(y_pred).argmax(axis=1)

result = pd.DataFrame(range(1, 28001), columns=['ImageId'])
result = result.assign(label=y_pred)

result.to_csv("digit-recognizer/data/submit.csv", index=False)
