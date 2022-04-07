import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 

from tensorflow import keras 
from tensorflow.keras.callbacks import EarlyStopping

from keras import layers
from keras.layers import Dropout,Activation
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

#mshahulh 07/04/2022 
import soundfile

DATASET_PATH = 'C:\Ai_audio\\Audio\\'
   
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
    
    
    header = 'filename'
    extraction = 'mfcc melspec chroma_stft chroma_cq chroma_cens'
    extraction = extraction.split()
    print(len(extraction))
    for ext in extraction:
        for i in range(1, 37):
            header += f' {ext}{i}'
            print(header)
    header += ' label'
    header = header.split()
    file = open('dataset(5_feature).csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for filename in os.listdir(DATASET_PATH):   
        for test in os.listdir(DATASET_PATH + "\\" + filename):
            for final in  os.listdir(DATASET_PATH + "\\" + filename + "\\"+ test):
                songname = os.path.join(DATASET_PATH,filename,test,final)
                print("loop ", songname)
                
                y, sr = librosa.load(songname, mono = True, duration = 30)
                mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
                melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
                chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
                chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
                chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
                features=np.hstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens))
                to_append = f'{filename}'
                for e in features:
                    to_append += f' {e}'   
                to_append += f' {test}'
                file = open('dataset(5_feature).csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())
                   
        print("complete loop: ",filename)
    return y,sr

def dataPreprocessing():    
    data = pd.read_csv('dataset(5_feature).csv')
    data.head()# Dropping unneccesary columns
    data = data.drop(['label'],axis=1)#Encoding the Labels
    genre_list = data.iloc[:,0]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)#Scaling the Feature columns
    print(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, 1:], dtype = float))#Dividing data into training and Testing set
  


    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
    print(X_train)
    print(y_train)
    print(X_test.shape)
    print(y_test.shape)
    # np.savetxt("test_data.csv",X_test,delimiter=",")
    # np.savetxt("train_labels.csv",y_train,delimiter=",")
    # np.savetxt("test_labels.csv",y_test,delimiter=",")
    
    
    
    return X_train,y_train,X_test,y_test

def AITraining(Ai_in,Ai_out,X_test,y_test):
    X_train = Ai_in
    print("X_train")
    print(X_train.shape[1])
    y_train = Ai_out
    model = Sequential()
    
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    
    model.add(layers.Dense(30, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(layers.Dense(12, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    # model.add(layers.Dense(12, activation='relu',kernel_regularizer = keras.regularizers.l2(0.001)))
    # model.add(Dropout(0.3))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    print(model.summary())
    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=4 )
    # classifier = model.fit(X_train,
                        # y_train,
                        # validation_data =(X_test,y_test),
                        # epochs=100,
                        # batch_size=32,
                        # callbacks=[early_stopping])
                        
    classifier = model.fit(X_train,
                        y_train,
                        validation_data =(X_test,y_test),
                        epochs=100,
                        batch_size=32)
  

    plot_history(classifier)    
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
    
# dataextraction()
trainIn,trainOut,testIn,testOut=dataPreprocessing()
AITraining(trainIn,trainOut,testIn,testOut)

# pitch_scale(signal, sr, -4, "tryag","2")

#07/04/2022 mshahulh
def generate_saturated(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050, mono=True)
    audio = audio * 5 #can try to play with the multiplier to amplify/minimize the signal
    soundfile.write(audio_file, data=audio, samplerate=22050)

def make_distortion_data():
    arr_features=[]
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        if genre == 'blues': #ToDo complete this list
            for fname in os.listdir(genre):
                generate_oversaturated(genre+'/'+fname)  

if __name__=='__main__': 
    make_distortion_data()

