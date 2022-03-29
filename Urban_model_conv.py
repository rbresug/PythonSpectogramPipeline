
#extracting data from csv files into numpy arrays
from numpy import genfromtxt
x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')

#shape
x_train.shape,x_test.shape,y_train.shape,y_test.shape



from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_train.shape,y_test.shape

from keras import Sequential
from keras.layers import Dense,Dropout,Activation

#forming model
model=Sequential()


#building the model
model.add(Dense(units=256,activation='relu',input_dim=200))
model.add(Dropout(0.4))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=10,activation='softmax'))

#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting
model.fit(x_train,y_train,epochs=30,validation_data=(x_test,y_test),batch_size=50)

#train and test loss and scores respectively
train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)

