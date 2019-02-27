#import libraries

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, BatchNormalization, Activation, Dropout
from keras.optimizers import adam
from random import shuffle
import matplotlib as mpl
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras
import scipy.io as sio
from sklearn.model_selection import KFold

#load data from matlab file
mat_contents = sio.loadmat('datar_2class_interp.mat')
X = mat_contents['bigdata_r']
y = mat_contents['label_r']

#normalize pixel data
X = X.astype('float32') / 255

#define regularization and droprate hyperparameters
reg_param = 0.0015
droprate = 0.50

#build model
model = Sequential()

#convolution 1, with batch norm
model.add(Conv2D(filters=64,strides=(2, 2), use_bias=False, kernel_size=5,
                                 padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(reg_param),
                                 input_shape=(50,50,4)))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(droprate))


#convolution 2, with batch norm and max pooling
model.add(Conv2D(filters=128,strides=(2, 2), use_bias=False, kernel_size=5,
                                 padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(reg_param)))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(droprate))

#convolution 3, with batch norm and max pooling
model.add(Conv2D(filters=256,strides=(2, 2), use_bias=False, kernel_size=3,
                                 padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(reg_param)))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(droprate))


#dense layer
model.add(Flatten())
model.add(Dense(64,kernel_regularizer=tf.keras.regularizers.l2(reg_param)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(droprate))

model.add(Dense(2, activation='softmax'))

model.summary()

#compilation

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

#training

class_weight = {0: 2.,
                1: 1.}

#create array to store validation accuracy in each iteration
scores = []

#create KFold, run for loop for k iterations
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
    model.fit(X_train,y_train,
         batch_size=20,
         epochs=15,
         validation_data=(X_test, y_test),
         class_weight=class_weight,
         callbacks=[checkpointer])
    score = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    scores.append(score[1])

print("Mean Score: ", np.mean(scores))
