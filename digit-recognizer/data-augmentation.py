RANDOM_SEED = 42

import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import RandomRotation, RandomZoom, Input, Flatten

train = pd.read_csv('digit-recognizer/data/train.csv')
X_train = train.iloc[:,1:]
y_train = np.array(train.iloc[:,0]).reshape(-1, 1)

X_train = np.array(X_train).reshape((-1, 28, 28))

aug = None

for i in range(2):
    augmentation_layer = Sequential()
    augmentation_layer.add(Input(shape=(28, 28, 1)))
    augmentation_layer.add(RandomRotation(factor=.1 * i, seed=RANDOM_SEED, fill_mode="constant", fill_value=0.))
    augmentation_layer.add(RandomZoom(height_factor=.1 * i, width_factor=.1 * i, fill_mode="constant", fill_value=0.))
    augmentation_layer.add(Flatten())
    x_aug = augmentation_layer(X_train, training=True)
    x_aug = np.hstack((y_train, x_aug))
    if aug is None:
        aug = np.copy(x_aug)
    else:
        aug = np.append(aug, x_aug, axis=0)


cols = ["label"]
cols = cols.append([f"pixel{i}" for i in range(784)])

result = pd.DataFrame(aug, columns=cols)
print(result.info())
result.to_csv("digit-recognizer/data/aug_train.csv", index=False)