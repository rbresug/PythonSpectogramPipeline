
#extracting data from csv files into numpy arrays
from numpy import genfromtxt
import pandas as pd


#y_train = genfromtxt('train_labels.csv', delimiter=',')
cols = genfromtxt('train_data.csv', delimiter=',')
print(cols)
x_train= cols[:, :-1] # for all but last column
y_train = cols[:, -1] # for last column



colstest = genfromtxt('test_data.csv', delimiter=',')
print(cols)
x_test= colstest[:, :-1] # for all but last column
y_test = colstest[:, -1] # for last column

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

from keras import Sequential
from keras.layers import Dense,Dropout,Activation

#forming model
model=Sequential()


#building the model
model.add(Dense(units=256,activation='relu',input_dim=169))
model.add(Dropout(0.4))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=10,activation='softmax'))

#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting
import tensorflow as tf

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
valid_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model.fit(x_train,y_train,epochs=30,validation_data=(x_test,y_test),batch_size=50)
#model.fit(train_data,epochs=30,validation_data=valid_data,batch_size=50)

#train and test loss and scores respectively
train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)

