import librosa
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import soundfile as sf
import os
from PIL import Image
import pathlib
import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import confusion_matrix

from collections import OrderedDict

from tensorflow import keras 
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# import openvino_ensorflow

from keras.utils.np_utils import to_categorical
from keras import layers
from keras.layers import Dropout,Activation,Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
print("can")
 
original_path = os.getcwd()

genres = 'NoerrorReal NoerrorArtificial Silence Robot'.split()
i = genres[0] 

def loadaudio():
    audioData = []
    for filename in os.listdir(DATASET_PATH):
        for test in os.listdir(DATASET_PATH + "\\" + filename):
            for final in  os.listdir(DATASET_PATH + "\\" + filename + "\\"+ test):
                filePath = os.path.join(DATASET_PATH,filename,test,final)
                audioData.append(filePath)
                
    return audioData,filename,test





def pitch_scale(signal, sr, num_semitones, name, i):
    agumented_y =librosa.effects.pitch_shift(signal,sr,num_semitones)
    sf.write('C:\\Ai_audio\\Audio\\Noerror\\Agumented\\agumentedReal_{name}_{i}.wav'.format(name=name,i=i),agumented_y,sr)
    # print(filePath)
    
    
def dataextraction(): 
    
    
    # header = 'filename'
    # extraction = 'mfcc melspec chroma_stft chroma_cq chroma_cens'
    # extraction = extraction.split()
    # print(len(extraction))
    # for ext in extraction:
        # for i in range(1, 37):
            # header += f' {ext}{i}'
            # # print(header)
    # header += ' label'
    # header = header.split()
    # file = open('dataset.csv', 'w', newline='')
    # with file:
        # writer = csv.writer(file)
        # writer.writerow(header)
        
    os.chdir('Audio')
    DATASET_PATH = os.getcwd()
    for filename in tqdm(os.listdir(DATASET_PATH),total = len(os.listdir(DATASET_PATH))):   
        for test in os.listdir(DATASET_PATH + "\\" + filename):
            for final in  os.listdir(DATASET_PATH + "\\" + filename + "\\"+ test):
                songname = os.path.join(DATASET_PATH,filename,test,final)
                print("loop ", songname)
                # print(original_path)
                
                y, sr = librosa.load(songname, duration = 30)
                mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
                melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
                chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
                chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
                chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
                features=np.hstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens))
                # to_append = f'{filename}'
                # # kprint(to_append)
                # # print(features)
                # # print(len(features))
                # # print(mfccs)
                # # print(len(mfccs))
                # # for e in mfccs:
                    # # to_append += f' {e}'
                # for e in features:
                    # to_append += f' {e}'   
                # to_append += f' {filename}'
                # file = open(f'{original_path}\dataset.csv', 'a', newline='')
                # with file:
                    # writer = csv.writer(file)
                    # writer.writerow(to_append.split())
                   
        # print("complete loop: ",filename)# 
    # # # # i -= 1
        # # print(i)
    
    # return y,sr

def dataPreprocessing():
    os.chdir(original_path)
    data = pd.read_csv('dataset.csv')
    # print(data)
    data.head()# Dropping unneccesary columns
    data = data.drop(['label'],axis=1)#Encoding the Labels
    # print(data)
    genre_list = data.iloc[:,0]
    # print(genre_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)#Scaling the Feature columns
    # print("label shape: " + type(y))
    # scaler = StandardScaler()
    # print(np.array(data.iloc[:,1:-1]))
    # X = scaler.fit_transform(np.array(data.iloc[:, 1:], dtype = float))#Dividing data into training and Testing set
    X = np.array(data.iloc[:, 1:], dtype = float)#Dividing data into training and Testing set
    print(X)
    # print(type(X))
    # print(X.dtype)
    # print(X.shape)

    # print(len(X))
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
    print(X_train)
    # print(X_test.shape)
    print(y_train)
    print(X_test.shape)
    print(y_test.shape)
    np.savetxt("train_data.csv",X_train,delimiter=",")
    np.savetxt("test_data.csv",X_test,delimiter=",")
    np.savetxt("train_labels.csv",y_train,delimiter=",")
    np.savetxt("test_labels.csv",y_test,delimiter=",")
    
    
    
    # return X_train,y_train,X_test,y_test
def dataPreprocessing2():
    os.chdir(original_path)
    data = pd.read_csv('dataset.csv')
    # print(data)
    data.head()# Dropping unneccesary columns
    data = data.drop(['label'],axis=1)#Encoding the Labels
    # print(data)
    genre_list = data.iloc[:,0]
    # print(genre_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)#Scaling the Feature columns
    # print("label shape: " + type(y))
    # scaler = StandardScaler()
    # print(np.array(data.iloc[:,1:-1]))
    # X = scaler.fit_transform(np.array(data.iloc[:, 1:], dtype = float))#Dividing data into training and Testing set
    X = np.array(data.iloc[:, 1:], dtype = float)#Dividing data into training and Testing set
    print(X)
    # print(type(X))
    # print(X.dtype)
    # print(X.shape)

    # print(len(X))
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)    
    return X_train,y_train,X_test,y_test

# def AITraining(Ai_in,Ai_out,X_test,y_test):
def AITraining():
    x_train = genfromtxt('train_data.csv', delimiter=',')
    y_train = genfromtxt('train_labels.csv', delimiter=',')
    x_test = genfromtxt('test_data.csv', delimiter=',')
    y_test = genfromtxt('test_labels.csv', delimiter=',')
    
    
    # X_train = Ai_in
    # print("X_train")
    # print(X_train.shape[1])
    # y_train = Ai_out
    
    # print(X_train.shape)
    # print(X_test.shape)
    
    y_train = to_categorical(y_train, num_classes=6)
    y_test = to_categorical(y_test, num_classes=6)
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

    print(x_train.shape)
    print(x_test.shape)
    
    model = Sequential()
############################################################################################################3 
    # model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    
    # model.add(layers.Dense(512, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # model.add(layers.Dense(256, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # # model.add(layers.Dense(12, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001)))
    # # model.add(Dropout(0.3))
    
    # model.add(layers.Dense(6, activation='softmax'))
    
    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
################################################################################################################3

    model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(36,5,1)))
    model.add(MaxPooling2D(padding="same"))

    model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
    model.add(MaxPooling2D(padding="same"))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256,activation="relu",))
    model.add(Dropout(0.3))

    model.add(Dense(128,activation="relu", ))
    model.add(Dropout(0.3))

    model.add(Dense(6,activation="softmax"))

#compiling
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=4 )
    # classifier = model.fit(x_train,
                        # y_train,
                        # validation_data =(x_test,y_test),
                        # epochs=100,
                        # batch_size=32,
                        # callbacks=[early_stopping])
    classifier = model.fit(x_train,
                        y_train,
                        validation_data =(x_test,y_test),
                        epochs=35,
                        batch_size=50)
  
    train_loss_score=model.evaluate(x_train,y_train)
    test_loss_score=model.evaluate(x_test,y_test)
    
    print(train_loss_score)
    print(test_loss_score)
    print(model.summary())
    
    plot_history(classifier)    
    
   

    #Generate the confusion matrix
    preds = np.argmax(model.predict(x_test), axis = 1)
    y_orig = np.argmax(y_test, axis = 1)
    cm = confusion_matrix(preds, y_orig)
    
    print(cm)
    keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()

    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, keys, normalize=True)
    plt.show()
    
    
    
def AITraining2(Ai_in,Ai_out,X_test,y_test):
   
    
    X_train = Ai_in
    print("X_train")
    print(X_train.shape[1])
    y_train = Ai_out
    
    print(X_train.shape)
    print(X_test.shape)
    
    model = Sequential()
############################################################################################################3 
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    
    model.add(layers.Dense(512, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(layers.Dense(256, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    # model.add(layers.Dense(12, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.3))
    
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
################################################################################################################3

    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=4 )
    # classifier = model.fit(x_train,
                        # y_train,
                        # validation_data =(x_test,y_test),
                        # epochs=100,
                        # batch_size=32,
                        # callbacks=[early_stopping])
    classifier = model.fit(X_train,
                        y_train,
                        validation_data =(X_test,y_test),
                        epochs=35,
                        batch_size=50)
  
    train_loss_score=model.evaluate(X_train,y_train)
    test_loss_score=model.evaluate(X_test,y_test)
    
    print(train_loss_score)
    print(test_loss_score)
    print(model.summary())
    
    plot_history(classifier)    
    
   

    #Generate the confusion matrix
    preds = np.argmax(model.predict(X_test))
    y_orig = np.argmax(y_test)
    cm = confusion_matrix(preds, y_orig)
    
    print(cm)
    
    # model.save("AI_audio(5_feature)")
    # print("model save")
    # test_loss_score=model.evaluate(X_test,y_test)
    # # print(train_loss_score)
    # print(test_loss_score)


def plot_history(history):
    fig,axs = plt.subplots(2)
    
    axs[0].plot(history.history["accuracy"],label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"],label = "test accuracy")
    axs[0].set_ylabel("Accuracry")
    axs[0].legend(loc = "lower right")
    axs[0].set_title("Accuracy eval")
    
    axs[1].plot(history.history["loss"],label = "train error")
    axs[1].plot(history.history["val_loss"],label = "test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc = "upper right")
    axs[1].set_title("Error eval")
    
    plt.show()
    
dataextraction()
# trainIn,trainOut,testIn,testOut=dataPreprocessing()
# AITraining()

# trainIn,trainOut,testIn,testOut=dataPreprocessing2()
# AITraining2(trainIn,trainOut,testIn,testOut)

# pitch_scale(signal, sr, -4, "tryag","2")


