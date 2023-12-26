RANDOM_SEED = 42

import keras
import matplotlib.pyplot as plt
from keras import Sequential
import numpy as np
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('digit-recognizer/data/aug_train.csv')
X_test = pd.read_csv('digit-recognizer/data/test.csv')
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

# Normalize to floats in [0:1] --> Why does this significantly improve performance?
X_train = X_train.astype('float32') / 255

print(f"Training set: {X_train.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

# Reshape to 28 x 28 pixel images
X_train = np.array(X_train).reshape((-1, 28, 28))
X_test = np.array(X_test).reshape((-1, 28, 28))

# Hot encode labels
y_train = np.array(y_train).reshape(-1, 1)
y_train = OneHotEncoder(sparse=False).fit_transform(y_train)

# Show image
# plt.subplot(2,2,1)
# plt.imshow(X_train[0 * 42000 + 42])
# plt.subplot(2,2,2)
# plt.imshow(X_train[1 * 42000 + 42])
# plt.subplot(2,2,3)
# plt.imshow(X_train[2 * 42000 + 42])
# plt.subplot(2,2,4)
# plt.imshow(X_train[3 * 42000 + 42])
# plt.show()
# plt.imshow
# print(y_train[42])
# exit()

model = Sequential()
model.add(Conv2D(32,(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(rate=0.15))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=20, validation_split=.1)

model.save("digit-recognizer/model.keras")

y_pred = model.predict(X_test)

y_pred = np.array(y_pred).argmax(axis=1)

result = pd.DataFrame(range(1, 28001), columns=['ImageId'])
result = result.assign(label=y_pred)

result.to_csv("digit-recognizer/data/submit.csv", index=False)
