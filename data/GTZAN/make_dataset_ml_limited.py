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


from numpy.linalg import norm

from dtw import dtw


#mshahulh 07/04/2022 
import soundfile
from Audio_model_cnn_conv_limited import *

from tensorflow.keras.models import Model,load_model

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
    
    num_clicks = 20
    # I purposely generate random int between 2 - num_clicks to varies the location of the click n pops in the audio
    length_vals = []
    amp_vals = []
    for x in range(num_clicks):
        rand_length = rd.randint(2,num_clicks)
        rand_amp = rd.randint(5,50)
        length_vals.append(rand_length)
        amp_vals.append(float(rand_amp) / 100)
    for click_loc, amp in zip(length_vals,amp_vals):
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

#07/04/2022 mshahulh
def generate_saturated(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050, mono=True)
    audio = audio * 5 #can try to play with the multiplier to amplify/minimize the signal
    soundfile.write(audio_file, data=audio, samplerate=22050)


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
    os.chdir('distort')
    #os.chdir('train')
    distort = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()



    for idx,distort in tqdm(enumerate(distort),total=len(distort)):
        for fname in os.listdir(distort):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
            y, sr = librosa.load(distort+'/'+fname, duration=10)
            print(distort+'/'+fname)
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
    orig_path = "distort_orig/"
    cd = os.getcwd()
    print ("curdir", cd)
    os.chdir('distort')
    #os.chdir('train')
    distort = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    #distort = 'blues'.split()
    for idx,distort in tqdm(enumerate(distort),total=len(distort)):
        for fname in os.listdir(distort):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
            y, sr = librosa.load(distort+'/'+fname, duration=30)

            y2, sr2 = librosa.load('../'+ orig_path + distort+'/'+fname, duration=30)

            mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
            mfccs2 = np.mean(librosa.feature.mfcc(y2, sr2, n_mfcc=36).T,axis=0)

            mfccs_diff = mfccs2 - mfccs
            print(mfccs_diff.shape)
            # mfcc1 = librosa.feature.mfcc(y,sr)   #Computing MFCC values
            # mfcc2 = librosa.feature.mfcc(y2, sr2)
            # dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))


            melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
            chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)

            # w = (mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)
            # print(w)
            # print("size", len(mfccs) + len(melspectrogram)+ len(chroma_stft) + len(chroma_cq) + len(chroma_cens) )
            # print("size", len(w))
            # print("comparison... ")
            # w2 = (dist.flatten(), cost.flatten(), acc_cost.flatten(), path.flatten())
            # print("dist",w2.shape)
            # print("cost",cost)
            # print("acc_cost",acc_cost)
            # print("path",path)
            # #print("size", len(dist) + len(cost)+ len(chroma_stft) + len(acc_cost) + len(path) )
            # print("size", len(w2))
            
            
            # features2=np.reshape(np.vstack((dist, cost, acc_cost, path)),(36,5))

            features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens,mfccs_diff)),(36,6))
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
    os.chdir('distort')
    orig_path = "distort_orig/"
    distort = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for fnamed in tqdm(os.listdir('test'),total=10*len(distort)):
        for fname in os.listdir('test/'+fnamed):
            print('test/'+ fnamed +'/'+fname)
            y, sr = librosa.load('test/'+ fnamed +'/'+fname, duration=30)
            y2, sr2 = librosa.load('../'+ orig_path + 'test/'+ fnamed +'/'+fname, duration=30)
            mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
            mfccs2 = np.mean(librosa.feature.mfcc(y2, sr2, n_mfcc=36).T,axis=0)

            mfccs_diff = mfccs2 - mfccs
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
            chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
            chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
            features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens, mfccs_diff)),(36,6))
            x_test.append(features)
            print("index", distort.index(fnamed.split('.')[0]))
            y_test.append(distort.index(fnamed.split('.')[0]))

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
    os.chdir('distort')
    #os.chdir('train')
    #distort = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    distort = 'blues'.split()
    for idx,distort in tqdm(enumerate(distort),total=len(distort)):
        for fname in os.listdir(distort):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
            y, sr = librosa.load(distort+'/'+fname, duration=10)
            print(distort+'/'+fname)
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


def make_distortion_data_all():
    os.chdir('distort')
    make_distortion_current_folder()
    cd = os.getcwd()
    print ("curdir", cd)
    os.chdir('test')
    make_distortion_current_folder()
    os.chdir('..')
    cd = os.getcwd()
    print ("curdir", cd)
    os.chdir('..')

def make_distortion_current_folder():
    #os.chdir('train')
    distort = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for idx,distort in tqdm(enumerate(distort),total=len(distort)):
        if distort == 'randomsilence': #ToDo complete this list
            for fname in os.listdir(distort):
                
                generate_randomsilence(distort+'/'+fname)        
        if distort == 'click_n_pop':
            for fname in os.listdir(distort):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_click_n_pops(distort+'/'+fname)
        if distort == 'discontinuity':
            for fname in os.listdir(distort):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_discontinuity(distort+'/'+fname)
        if distort == 'hum':
            for fname in os.listdir(distort):
                generate_hum(distort+'/'+fname)
        if distort == 'white_noise2':
            for fname in os.listdir(distort):
                generate_white_noise2(distort+'/'+fname)
        if distort == 'hum2':
            for fname in os.listdir(distort):
                generate_hum2(distort+'/'+fname) 
        if distort == 'oversampling':
            for fname in os.listdir(distort):
                generate_oversampling(distort+'/'+fname) 
        if distort == 'undersampling':
            for fname in os.listdir(distort):
                generate_undersampling(distort+'/'+fname) 
        if distort == 'saturated':
            for fname in os.listdir(distort):
                generate_saturated(distort+'/'+fname) 
        if distort == 'randomsilence5':
            for fname in os.listdir(distort):
                generate_randomsilence5(distort+'/'+fname)
        

def make_test_data():
    arr_features=[]
    os.chdir('distort')
    #distort = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    distortions = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for fname in tqdm(os.listdir('test'),total=10*len(distort)):
            y, sr = librosa.load('test/'+fname, duration=30)
            dict_features=extract_features(y=y,sr=sr)
            dict_features['label']=distort.index(fname.split('.')[0])
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    print(df.head())
    print(df.shape)
    os.chdir('..')
    df.to_csv('test_data.csv',index=False)



#this function will check the content of folder test ( checks only files and ignores directories) and will just 
def make_test_data_no_label():

    # File path
    filepath_model = './custom_cnn_2d.h5'

    # Load the model
    model = load_model(filepath_model, compile = True)

    x_test=[]
    y_test=[]
    arr_features=[]
    os.chdir('distort')
    orig_path = "distort_orig/"
    #             0             1           2           3       4       5       6           7               8             9
    distort = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    
    test_files = [f for f in os.listdir('test') if os.path.isfile(os.path.join('test', f))]
    print("aaa", test_files)
    #for fname in tqdm(os.listdir('test'),total=10*len(distort)):
    for fname in test_files:
        print('test/'+fname)
        y, sr = librosa.load('test/'+fname, duration=30)
        #y2, sr2 = librosa.load('../'+ orig_path + 'test/'+  fname, duration=30)
        y2, sr2 = librosa.load('test/'+fname, duration=30)
        mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
        mfccs2 = np.mean(librosa.feature.mfcc(y2, sr2, n_mfcc=36).T,axis=0)

        mfccs_diff = mfccs2 - mfccs
        #TODO when files are the same ( e.g. positive test result, and no distortion), then the diff vector will be null
        # if ... mfccs_diff == 0
        melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
        chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
        features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens, mfccs_diff)),(36,6))
        x_test.append(features)
        ##print("index", distort.index(fnamed.split('.')[0]))
        y_test.append(0)
        print("tempo_dbg",    features.shape)
        # Convert into Numpy array
        samples_to_predict = np.array(features)
        # print(samples_to_predict.shape)
        # samples_to_predict=np.reshape(samples_to_predict,(samples_to_predict.shape[0], 36,6))
        # print(samples_to_predict.shape)

        samples_to_predict=np.reshape(samples_to_predict,(1, 36,6,1))

        # Generate predictions for samples
        preds = model.predict(samples_to_predict, batch_size = 128, verbose= 1)
        #predictions = model.predict(samples_to_predict, batch_size = 128, verbose= 1)
        print("predictions orig", preds)
        classes = np.argmax(preds, axis = 1)
        print("classes orig", classes)

        pred=np.zeros(10)
        preds=np.array([preds[0,0],preds[0,1],preds[0,2],preds[0,3],preds[0,4],preds[0,5],preds[0,6], preds[0,7],preds[0,8],preds[0,9]])
        pred=(preds+(3*pred))/4
        print("predictions", pred)
        # Generate arg maxes for predictions
        #classes = np.argmax(predictions, axis = 1)
        #print("classes", classes)
        maximum = max(pred)
        print("maximumt", maximum)
        classes = np.where(pred == maximum)
        print("classes after different", classes)
        print("classes val after different", pred[classes])
        classes = np.argmax(pred, axis=0)
        print("classes afster", classes)



        ##cnn
    
    
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    os.chdir('..')
    np.savetxt("test_data_nolabel.csv",x_test_2d,delimiter=",")
    np.savetxt("test_labels_nolabel.csv",y_test,delimiter=",")
    



if __name__=='__main__':
    #ToDo insert feature of introducing distortions
    #Blues = random silence
    # Classical = Clicks and pops
    # Country = Discontinuity
    # Disco = Hum introduction
    # hiphop  = Inter-sample peaks 
    ##make_distortion_data_all()
    #make_train_data_column()
    #make_test_data_column()
    #run_nn()
    make_test_data_no_label()