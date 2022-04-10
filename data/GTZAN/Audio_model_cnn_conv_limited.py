
#extracting data from csv files into numpy arrays
from numpy import genfromtxt
import pandas as pd


import numpy as np

import itertools

import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm



#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def majority_vote(scores):
    print("scores: ", scores)
    values, counts = np.unique(scores,return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def run_nn():
    x_train = genfromtxt('train_data.csv', delimiter=',')
    y_train = genfromtxt('train_labels.csv', delimiter=',')
    x_test = genfromtxt('test_data.csv', delimiter=',')
    y_test = genfromtxt('test_labels.csv', delimiter=',')

    song_samples = 660000
    genres = {'blues': 0 ,'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
            'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

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
    x_train=np.reshape(x_train,(x_train.shape[0], 36,6))
    x_test=np.reshape(x_test,(x_test.shape[0], 36,6))
    x_train.shape,x_test.shape



    #reshaping to shape required by CNN
    x_train=np.reshape(x_train,(x_train.shape[0], 36,6,1))
    x_test=np.reshape(x_test,(x_test.shape[0], 36,6,1))

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



    hist = model.fit(x_train,y_train,batch_size=50,epochs=30,validation_data=(x_test,y_test))
    #model.fit(train_data,epochs=30,validation_data=valid_data,batch_size=50)


    #train and test loss and scores respectively
    train_loss_score=model.evaluate(x_train,y_train)
    score=model.evaluate(x_test,y_test)
    print(train_loss_score)
    print(score)


    plt.figure(figsize=(15,7))
    print("hist.history  ", hist.history)
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('graph.png')





    preds = np.argmax(model.predict(x_test), axis = 1)
    y_orig = np.argmax(y_test, axis = 1)
    cm = confusion_matrix(preds, y_orig)




    keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, keys, normalize=True)
    plt.show()
    plt.savefig('plot_confusion_matrix.png')
    # ## Majority Vote






    preds = model.predict(x_test, batch_size=128, verbose=0)




    # Each sound was divided into 39 segments in our custom function
    scores_songs = np.array_split(np.argmax(preds, axis=1), 300)
    scores_songs = [majority_vote(scores) for scores in scores_songs if scores != []]




    # Same analysis for split
    label = np.array_split(np.argmax(y_test, axis=1), 300)
    label = [majority_vote(l) for l in label if l != []]




    from sklearn.metrics import accuracy_score

    print("majority voting system (acc) = {:.3f}".format(accuracy_score(label, scores_songs)))


    # Compared to the classical approach, we are winning now!
    # 

    # ## Save the model



    # Save the model
    model.save('custom_cnn_2d.h5')



