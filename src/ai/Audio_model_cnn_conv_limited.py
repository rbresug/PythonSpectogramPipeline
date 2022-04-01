
#extracting data from csv files into numpy arrays
from numpy import genfromtxt
import pandas as pd


import numpy as np






x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')



''' cols2 = pd.read_csv('test_data.csv', nrows=1).columns
x_test = pd.read_csvt('test_data.csv', delimiter=',')#, usecols=cols2[:-1])
y_test = pd.read_csv('test_data.csv', delimiter=',', usecols=cols2[-1])
print ("x_test: ", x_test)
print ("y_test: ", y_test) '''



#shape
x_train.shape,x_test.shape,y_train.shape,y_test.shape


#import tensorflow as tf
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_train.shape,y_test.shape
##cnn
#reshaping to 2D 
x_train=np.reshape(x_train,(x_train.shape[0], 36,5))
x_test=np.reshape(x_test,(x_test.shape[0], 36,5))
x_train.shape,x_test.shape



#reshaping to shape required by CNN
x_train=np.reshape(x_train,(x_train.shape[0], 36,5,1))
x_test=np.reshape(x_test,(x_test.shape[0], 36,5,1))

x_train.shape,x_test.shape
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

#forming model
model=Sequential()


#building the model
#adding layers and forming the model
model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(36,5,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(10,activation="softmax"))

#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting
import tensorflow as tf



model.fit(x_train,y_train,batch_size=50,epochs=30,validation_data=(x_test,y_test))
#model.fit(train_data,epochs=30,validation_data=valid_data,batch_size=50)



#train and test loss and scores respectively
train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)



