#
#for 2- class problem only
#
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from random import shuffle
from copy import deepcopy
#
#read matlab file
#
mat_contents = sio.loadmat('datar_2class_interp.mat')
bdata = mat_contents['bigdata_r']
lab = mat_contents['label_r']
print(bdata.shape)
print(lab.shape)
#
#convert to numpy arrays 
#
bdata = np.array(bdata)
lab = np.array(lab)
print(bdata.shape)
print(lab.shape)
#
#normalize data
#
bdata = bdata.astype('float32') / 255
#define training and validation data 
#
num_valid = 15*16
num_total = 143*16
num_train = num_total - num_valid
ivalid_s =num_valid*0
#ivalid_s = num_total - num_valid
ivalid_e = ivalid_s + num_valid
print(ivalid_e,'ivalid_end')
print(ivalid_s,'ivalid_start')
#
x_train = deepcopy(bdata[:num_train])
x_train[:ivalid_s] = deepcopy(bdata[:ivalid_s])
x_train[ivalid_s:] = deepcopy(bdata[ivalid_e:])
y_train = deepcopy(lab[:num_train])
y_train[:ivalid_s] = deepcopy(lab[:ivalid_s])
y_train[ivalid_s:] = deepcopy(lab[ivalid_e:])
#
x_valid = deepcopy(bdata[ivalid_s:ivalid_e])
y_valid = deepcopy(lab[ivalid_s:ivalid_e])
# Print the number of training and validation sets
print(x_train.shape, 'train set images')
print(x_valid.shape, 'validation set images')
print(y_train.shape, 'train set labels')
print(y_valid.shape, 'validation set labels')
#
# some checks 
#
## print the first training sequence 
#np.savetxt('train01.out', x_train[0,:,:,2], delimiter=',') 
#i1  = 0
#if ivalid_s == 0:
#  i1 = ivalid_e
#np.savetxt('orig_train01.out', bdata[i1,:,:,2], delimiter=',') 
##
## print the last training sequence 
#np.savetxt('train02.out', x_train[num_train-1,:,:,2], delimiter=',') 
#i1  = num_total
#if ivalid_e == num_total:
#  i1 = num_total - num_valid 
#np.savetxt('orig_train02.out', bdata[i1-1,:,:,2], delimiter=',') 
##
# model starts here 
#
#reg_param = 0.0015
reg_param = 0.0
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides = (2,2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_param),input_shape=(50,50,4)))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
#model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', strides = (2,2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_param)))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
#model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides = (2,2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_param)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.3))

#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides = (2,2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_param)))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.5))

#model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides = (2,2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_param)))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_param)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax', ))

# Take a look at the model summary
model.summary()

model.compile(loss='mse',
#model.compile(loss='categorical_crossentropy',
#             optimizer='adam',
             optimizer='sgd',
             metrics=['accuracy'])

#from tf.keras.callbacks import ModelCheckpoint

#checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
class_weight = {0: 2.,
                1: 1.}
#checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
# Evaluate the model on validation set
fl1=open('hist.out','a')
score_train = model.evaluate(x_train, y_train, verbose=0)
score_valid = model.evaluate(x_valid, y_valid, verbose=0)
initscore = np.zeros(4)
initscore = [score_train[0],score_train[1],score_valid[0],score_valid[1]]
#initscore[0] = score_train[0]
#initscore[1] = score_train[1]
#initscore[2] = score_valid[0]
#initscore[3] = score_valid[1]
temp = np.matrix(initscore)
np.savetxt(fl1,temp)
#np.savetxt(fl1,initscore,newline=',')
print('train score before training',score_train)
print('valid score before training',score_valid)
#
history = model.fit(x_train,
         y_train,
         batch_size=20,
         epochs=800,
         validation_data=(x_valid, y_valid),
         class_weight=class_weight)
#         callbacks=[checkpointer])

# list all data in history
np.savetxt(fl1,np.transpose([history.history['loss'], history.history['acc'], history.history['val_loss'], history.history['val_acc']]),delimiter=' ')

# Evaluate the model on test set
score = model.evaluate(x_valid, y_valid, verbose=0)
#model.metrics_names
print(score)
y_pred = model.predict(x_valid)

fl2=open('pred.out','a')
fl3=open('ansr.out','a')
np.savetxt(fl2, y_pred, delimiter=',') 
np.savetxt(fl3, y_valid, delimiter=',') 

