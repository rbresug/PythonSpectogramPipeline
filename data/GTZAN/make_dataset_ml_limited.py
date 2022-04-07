import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew
#from IPython.display import Audio
from essentia import Pool, db2amp
from essentia import array, instantPower
from essentia.standard import MonoLoader, MonoWriter, DiscontinuityDetector, FrameGenerator
from essentia.standard import (
    Windowing,
    Spectrum,
    FrameGenerator,
    MonoLoader,
    Clipper,
    HumDetector
)
from essentia import array
import matplotlib.pyplot as plt
import numpy as np
import random as rd

from Audio_model_cnn_conv_limited import *

def generate_discontinuity(audio_file):

    plt.rcParams["figure.figsize"] = (12, 9)

    sr = 22050

    audio = MonoLoader(filename=audio_file, sampleRate =sr)()

    jump_starts = np.array([len(audio) // 2, len(audio) // 2])
    ground_truth = jump_starts / sr

    try:
        for start in jump_starts:
            l_amp = audio[start]
            # Remove samples until the jump produces a prominent discontinuity so it can be perceived.
            end = next(
                idx for idx, r_amp in enumerate(audio[start:], start) if abs(r_amp - l_amp) > 0.3
            )
            audio = np.delete(audio, range(start, end))
    except StopIteration:
        pass
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

def generate_randomsilence(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    
    
    time_axis = np.arange(len(audio)) / sr
    gap_position = rd.randint(2,28)
    gap_duration = 0.5
    gap_start = gap_position * sr
    gap_end = int(gap_start + gap_duration * sr)
    print(gap_duration * sr)
    
    audio[gap_start: gap_end] = np.zeros(int(gap_duration * sr))
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

def generate_undersampling(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    sr = sr / 2
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)


def generate_oversampling(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    sr = sr * 2
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)


def generate_randomsilence3(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    
    
    time_axis = np.arange(len(audio)) / sr
    gap_position = rd.randint(2,20)
    gap_duration = 3
    gap_start = gap_position * sr
    gap_end = int(gap_start + gap_duration * sr)
    print(gap_duration * sr)
    
    audio[gap_start: gap_end] = np.zeros(int(gap_duration * sr))
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

def generate_randomsilence4(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    
    time_axis = np.arange(len(audio)) / sr
    gap_position = rd.randint(2,20)
    gap_duration = 5
    gap_start = gap_position * sr
    gap_end = int(gap_start + gap_duration * sr)
    print(gap_duration * sr)
    
    audio[gap_start: gap_end] = np.zeros(int(gap_duration * sr))
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

def generate_randomsilence5(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
       
    time_axis = np.arange(len(audio)) / sr
    gap_position = rd.randint(2,20)
    gap_duration = 8
    gap_start = gap_position * sr
    gap_end = int(gap_start + gap_duration * sr)
    print(gap_duration * sr)
    
    audio[gap_start: gap_end] = np.zeros(int(gap_duration * sr))
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

#25/3/22 mshahulh
def generate_click_n_pops(audio_file):
    sr = 44100

    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    
    # I purposely generate random int between 2 - 5 to varies the location of the click n pops in the audio
    rand_length = []
    for x in range(3):
        y = rd.randint(2,5)
        rand_length.append(y)
        
    click_locs = [len(audio) // 4,    len(audio) // 2,    len(audio) * 3 // 4]

    for click_loc, amp in zip(click_locs, [.5, .15, .05]):#
        audio[click_loc] += amp
    
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)
#25/3/22 mshahulh
def generate_hum2(audio_file):
    
    sr = 44100
    
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()

    # Generate a 100Hz tone.
    time = np.arange(len(audio)) / sr
    freq = 100
    hum = np.sin(2 * np.pi * freq * time).astype(np.float32)

    # Add odd harmonics via clipping.
    hum = Clipper(min=-0.5, max=0.5)(hum)

    # Attenuate the hum 30 dB.
    hum *= db2amp(-30)

    audio_with_hum = audio + hum
    
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio_with_hum)

#25/3/22 mshahulh
def generate_hum(audio_file):
    
    sr = 44100
    
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()

    # Generate a 50Hz tone.
    time = np.arange(len(audio)) / sr
    freq = 30
    hum = np.sin(2 * np.pi * freq * time).astype(np.float32)

    # Add odd harmonics via clipping.
    hum = Clipper(min=-0.5, max=0.5)(hum)

    # Attenuate the hum 30 dB.
    hum *= db2amp(-30)

    audio_with_hum = audio + hum
    
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio_with_hum)
#28/3/2022 mshahulh
def generate_white_noise(audio_file):
    time = 25  # s   
    sr = 44100
    audio_db = -15
    noise_db = -95
    audio = MonoLoader(filename= audio_file, sampleRate= sr)()
    audio *= db2amp(audio_db)
    
    noise_only = 1



    time = np.arange(sr * time) / sr
    
    noise = np.random.randn(len(time)).astype(np.float32)
    noise *= db2amp(noise_db)

    audio_power = instantPower(audio)
    noise_power = instantPower(noise)
    true_snr = 10 * np.log10( audio_power / noise_power)

    print('audio level: {:.2f}dB'.format(10. * np.log10(audio_power)))
    print('Noise level: {:.2f}dB'.format(10. * np.log10(noise_power)))
    print('SNR: {:.2f}dB'.format(true_snr))
    if len(audio) < len(noise):
        c = noise.copy()
        c[:len(audio)] += audio
    else:
        c = audio.copy()
        c[:len(noise)] += noise
    MonoWriter(filename = audio_file ,format='wav',sampleRate=sr)(c)
def generate_white_noise2(audio_file):
    
    sr = 44100
    time = 25  # s
    audio_db = -10
    noise_db = -80
    audio = MonoLoader(filename= audio_file, sampleRate= sr)()
    audio *= db2amp(audio_db)
    
    noise_only = 1



    time = np.arange(sr * time) / sr
    
    noise = np.random.randn(len(time)).astype(np.float32)
    noise *= db2amp(noise_db)

    audio_power = instantPower(audio)
    noise_power = instantPower(noise)
    true_snr = 10 * np.log10( audio_power / noise_power)

    print('audio level: {:.2f}dB'.format(10. * np.log10(audio_power)))
    print('Noise level: {:.2f}dB'.format(10. * np.log10(noise_power)))
    print('SNR: {:.2f}dB'.format(true_snr))
    if len(audio) < len(noise):
        c = noise.copy()
        c[:len(audio)] += audio
    else:
        c = audio.copy()
        c[:len(noise)] += noise

    MonoWriter(filename = audio_file ,format='wav',sampleRate=sr)(c)


#23/03/22 RBresug:
#precondition is to download from wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
def extract_features(y,sr=22050,n_fft=1024,hop_length=512):
    features = {'centroid': librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'flux': librosa.onset.onset_strength(y=y, sr=sr).ravel(),
                'rmse': librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel(),
                'zcr': librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel(),
                'contrast': librosa.feature.spectral_contrast(y, sr=sr).ravel(),
                'bandwidth': librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'flatness': librosa.feature.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel(),
                'rolloff': librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()}

    # MFCC treatment
    mfcc = librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=20)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()

    # Get statistics from the vectors
    def get_feature_stats(features):
        result = {}
        for k, v in features.items():
            result['{}_max'.format(k)] = np.max(v)
            result['{}_min'.format(k)] = np.min(v)
            result['{}_mean'.format(k)] = np.mean(v)
            result['{}_std'.format(k)] = np.std(v)
            result['{}_kurtosis'.format(k)] = kurtosis(v)
            result['{}_skew'.format(k)] = skew(v)
        return result

    dict_agg_features = get_feature_stats(features)
    dict_agg_features['tempo'] = librosa.beat.tempo(y=y,sr=sr,hop_length=hop_length)[0]

    return dict_agg_features


#23/03/22 RBresug:
#precondition is to download from wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
def extract_features_limited(y,sr=22050,n_fft=1024,hop_length=512):
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(36,5))

    return features

def make_train_data():
    arr_features=[]
    os.chdir('genres')
    #os.chdir('train')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()



    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(genre):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
            y, sr = librosa.load(genre+'/'+fname, duration=10)
            print(genre+'/'+fname)
            print(y)
            print("size" + str(len(y)))
            print(sr)
            dict_features=extract_features_limited(y=y,sr=sr)
            dict_features['label']=idx
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    print(df.head())
    print(df.shape)
    os.chdir('..')
    df.to_csv('train_data.csv',index=False)

def make_train_data_column():
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    arr_features=[]
    cd = os.getcwd()
    print ("curdir", cd)
    os.chdir('genres')
    #os.chdir('train')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    #genres = 'blues'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(genre):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
            y, sr = librosa.load(genre+'/'+fname, duration=30)
            mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
            chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
            features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(36,5))
            x_train.append(features)
            y_train.append(idx)

    x_train=np.array(x_train)

    y_train=np.array(y_train)

    x_train_2d=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    os.chdir('..')
    np.savetxt("train_data.csv", x_train_2d, delimiter=",")
    np.savetxt("train_labels.csv",y_train,delimiter=",")

def make_test_data_column():
    x_test=[]
    y_test=[]
    arr_features=[]
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for fnamed in tqdm(os.listdir('test'),total=10*len(genres)):
        for fname in os.listdir('test/'+fnamed):
            print('test/'+ fnamed +'/'+fname)
            y, sr = librosa.load('test/'+ fnamed +'/'+fname, duration=30)
            mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
            chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
            features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(36,5))
            x_test.append(features)
            print("index", genres.index(fnamed.split('.')[0]))
            y_test.append(genres.index(fnamed.split('.')[0]))

    x_test=np.array(x_test)
    y_test=np.array(y_test)

    x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    os.chdir('..')
    np.savetxt("test_data.csv",x_test_2d,delimiter=",")
    np.savetxt("test_labels.csv",y_test,delimiter=",")


def make_train_data_orig():
    arr_features=[]
    cd = os.getcwd()
    print ("curdir", cd)
    os.chdir('genres')
    #os.chdir('train')
    #genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    genres = 'blues'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(genre):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
            y, sr = librosa.load(genre+'/'+fname, duration=10)
            print(genre+'/'+fname)
            print(y)
            print("size" + str(len(y)))
            print(sr)
            _features=extract_features_limited(y=y,sr=sr)
            arr_features.append(_features)
            arr_features.append(idx)

    df=pd.DataFrame(data=arr_features)
    print(df.head())
    print(df.shape)
    os.chdir('..')
    df.to_csv('train_data.csv',index=False)

def make_distortion_data():
    arr_features=[]
    os.chdir('genres')
    #os.chdir('train')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        
        if genre == 'blues': #ToDo complete this list
            for fname in os.listdir(genre):
                
                generate_randomsilence(genre+'/'+fname)        
        if genre == 'classical':
            for fname in os.listdir(genre):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_click_n_pops(genre+'/'+fname)
        if genre == 'country':
            for fname in os.listdir(genre):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                print("d")
                generate_discontinuity(genre+'/'+fname)
        if genre == 'disco':
            for fname in os.listdir(genre):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_hum(genre+'/'+fname)
        if genre == 'hiphop':
            for fname in os.listdir(genre):
                generate_white_noise2(genre+'/'+fname)
        if genre == 'jazz':
            for fname in os.listdir(genre):
                generate_hum2(genre+'/'+fname) 
        if genre == 'metal':
            for fname in os.listdir(genre):
                generate_oversampling(genre+'/'+fname) 
        if genre == 'pop':
            for fname in os.listdir(genre):
                generate_undersampling(genre+'/'+fname) 
        if genre == 'reggae':
            for fname in os.listdir(genre):
                generate_white_noise(genre+'/'+fname) 
        if genre == 'rock':
            for fname in os.listdir(genre):
                generate_randomsilence5(genre+'/'+fname)
    os.chdir('..')         

def make_test_data():
    arr_features=[]
    os.chdir('genres')
    #genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    distortions = 'gaps hums discontinuity disco hiphop jazz metal pop reggae rock'.split()
    for fname in tqdm(os.listdir('test'),total=10*len(genres)):
            y, sr = librosa.load('test/'+fname, duration=30)
            dict_features=extract_features(y=y,sr=sr)
            dict_features['label']=genres.index(fname.split('.')[0])
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    print(df.head())
    print(df.shape)
    os.chdir('..')
    df.to_csv('test_data.csv',index=False)



if __name__=='__main__':
    #ToDo insert feature of introducing distortions
    #Blues = random silence
    # Classical = Clicks and pops
    # Country = Discontinuity
    # Disco = Hum introduction
    # hiphop  = Inter-sample peaks 
    make_distortion_data()
    make_train_data_column()
    make_test_data_column()
    run_nn()
