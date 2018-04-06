__author__ = "NF"

# Use CNN for image classification

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils

# File I/O
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "data")
train_data_path = os.path.join(data_dir, 'train.csv')
test_data_path = os.path.join(data_dir, 'test.csv')
# Since data is csv use pandas to read it in
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

train_df.head(n=1)
train_df.shape
train_df.columns.values
# ** There's 42000 samples
n_samples, _ = train_df.shape

X_train = train_df.drop("label", axis=1)
y_train  = train_df["label"]
X_test = test_df
# Image must be repackaged into 2D, since each row represents unrolled image sqrt(783) = 28x28 resolution
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

# Prep data for Keras with Tensorflow backend
X_train = X_train.reshape(n_samples, 28, 28, 1)
X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Show an example
plt.imshow(X_train[1,:,:].reshape(28,28), cmap=plt.cm.gray)
plt.imshow(X_test[1,:,:].reshape(28,28), cmap=plt.cm.gray)

# Normalize the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_train[1,:]

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout

cnn = Sequential()
cnn.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5)) # represents the image template
cnn.add(Dense(10,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1)

# Final Predictions
y_pred = cnn.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

plt.imshow(X_test[1000,:,:,:].reshape(28,28))
plt.title(str(y_pred[1000]))

submission = pd.DataFrame({ "ImageId": list(range(1,len(y_pred)+1)),
                            "Label": y_pred})

submission.to_csv('./submission.csv', index=False)
