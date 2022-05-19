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
from pydub import AudioSegment
import array,wave

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
# print("can")
 
original_path = os.getcwd()





def dataextractionStereo(): 
  
    header = 'filename'
    extraction = 'mfcc melspectrogram chroma_stft chroma_cq chroma_cens'
    extraction = extraction.split()
    print(len(extraction))
    for ext in extraction:
        for i in range(1, 37):
            header += f' {ext}{i}'
            # print(header)
    header += ' label'
    header = header.split()
    file = open('dataset_TEST.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    os.chdir('Audio')
    DATASET_PATH = os.getcwd()
    for filename in tqdm(os.listdir(DATASET_PATH),total = len(os.listdir(DATASET_PATH))):   
        for test in os.listdir(DATASET_PATH + "\\" + filename):
            for final in  os.listdir(DATASET_PATH + "\\" + filename + "\\"+ test):
                audiopath = os.path.join(DATASET_PATH,filename,test,final)
                audiofile, sr = librosa.load(audiopath, duration=30,mono = False)
                for i in range (0,audiofile.ndim):
                    mfccs= np.mean(librosa.feature.mfcc(audiofile[i], sr, n_mfcc=36).T,axis=0)
                    melspectrogram = np.mean(librosa.feature.melspectrogram(audiofile[i], sr=sr, n_mels=36,fmax=8000).T,axis=0)
                    chroma_stft=np.mean(librosa.feature.chroma_stft(audiofile[i], sr=sr,n_chroma=36).T,axis=0)
                    chroma_cq = np.mean(librosa.feature.chroma_cqt(audiofile[i], sr=sr,n_chroma=36).T,axis=0)
                    chroma_cens = np.mean(librosa.feature.chroma_cens(audiofile[i], sr=sr,n_chroma=36).T,axis=0)
                    features=np.hstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens))
                    to_append = f'{test}'
                    for e in features:
                        to_append += f' {e}'   
                    to_append += f' {filename}'
                    file = open(f'{original_path}\dataset_TEST.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                               
        print("complete loop")
        os.chdir(original_path)
        
def make_stereo(file1, output):
       
    ifile = wave.open(file1)
    print (ifile.getparams())
    # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel
    
    os.chdir("stereo")
    ofile = wave.open(output, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tostring())
    ofile.close()
    os.chdir(original_path)
                
                
if __name__ == '__main__':
    dataextractionStereo()