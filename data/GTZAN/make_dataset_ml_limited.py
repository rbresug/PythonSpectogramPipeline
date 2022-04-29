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
import logging
logger = logging.getLogger(__name__)
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
    logger.debug("Done for file {}".format(audio_file))

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
    logger.debug("Done for file {}".format(audio_file))

def generate_undersampling(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    sr = sr / 2
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)
    logger.debug("Done for file {}".format(audio_file))


def generate_oversampling(audio_file):
    sr = 44100
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    sr = sr * 2
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)
    logger.debug("Done for file {}".format(audio_file))


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
    logger.debug("Done for file {}".format(audio_file))

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
    logger.debug("Done for file {}".format(audio_file))

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
    logger.debug("Done for file {}".format(audio_file))

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
    logger.debug("Done for file {}".format(audio_file))
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
    logger.debug("Done for file {}".format(audio_file))

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
<<<<<<< Updated upstream
#28/3/2022 mshahulh
=======
    logger.debug("Done for file {}".format(audio_file))

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
    logger.debug("Done for file {}".format(audio_file))
    
>>>>>>> Stashed changes
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
    logger.debug("Done for file {}".format(audio_file))

#07/04/2022 mshahulh
def generate_saturated(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050, mono=True)
    audio = audio * 5 #can try to play with the multiplier to amplify/minimize the signal
    soundfile.write(audio_file, data=audio, samplerate=22050)
    logger.debug("Done for file {}".format(audio_file))

<<<<<<< Updated upstream
=======
def generate_chop_every_on_frame(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050, mono=True)
    chop_on_frames = 10
    chop_size = 1
    print('Chop slice of {} frames on every {}'.format(chop_size, chop_on_frames))
    r_buf = [[], []]
    chop_counter = 0
    chop_size_counter = 0

    for i in range(len(audio[0])):
        if chop_size_counter < chop_size and chop_counter >= chop_on_frames:
            chop_size_counter += 1
        else:    
            chop_counter = 0
            chop_size_counter = 0

        if chop_counter < chop_on_frames:
            r_buf[0].append(audio[0][i])
            r_buf[1].append(audio[1][i])
            chop_counter += 1
    
    soundfile.write(audio_file, data=r_buf, samplerate=22050)
    logger.debug("Done for file {}".format(audio_file))
    
>>>>>>> Stashed changes

#23/03/22 RBresug:
#precondition is to download from wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
def extract_features(y,sr=22050,n_fft=1024,hop_length=512):
    logger.info("Starting feature extraction")
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
        logger.info("Subfunction: Get Feature Stats")
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
    logger.info("Extracting Features in Limited Data")
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=36).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=36).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=36).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=36).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(36,5))

    return features

def make_train_data():
    logger.info("Starting to make train data")
    arr_features=[]
    os.chdir('genres')
    #os.chdir('train')
    genres = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()



    for idx,distort in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(genre):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
<<<<<<< Updated upstream
            y, sr = librosa.load(distort+'/'+fname, duration=10)
            print(distort+'/'+fname)
            print(y)
            print("size" + str(len(y)))
            print(sr)
=======
            y, sr = librosa.load(distort+'/'+fname, duration=30)
            logger.debug(distort+'/'+fname)
            logger.debug(y)
            logger.debug("size" + str(len(y)))
            logger.debug(sr)
>>>>>>> Stashed changes
            dict_features=extract_features_limited(y=y,sr=sr)
            dict_features['label']=idx
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    logger.debug(df.head())
    logger.debug(df.shape)
    os.chdir('..')
    df.to_csv('train_data.csv',index=False)

def make_train_data_column():
    logger.info("Make Train Data Column")
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    arr_features=[]
    orig_path = "genres_orig/"
    cd = os.getcwd()
    logger.debug("curdir", cd)
    os.chdir('genres')
    #os.chdir('train')
    genres = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    #genres = 'blues'.split()
    for idx,distort in tqdm(enumerate(genres),total=len(genres)):
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
    logger.info("Make Test Data Column")
    x_test=[]
    y_test=[]
    arr_features=[]
    os.chdir('genres')
    orig_path = "genres_orig/"
    genres = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for fnamed in tqdm(os.listdir('test'),total=10*len(genres)):
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
<<<<<<< Updated upstream
            print("index", genres.index(fnamed.split('.')[0]))
            y_test.append(genres.index(fnamed.split('.')[0]))
=======
            logger.debug("index", distortion_type.index(fnamed.split('.')[0]))
            y_test.append(distortion_type.index(fnamed.split('.')[0]))
>>>>>>> Stashed changes

    x_test=np.array(x_test)
    y_test=np.array(y_test)

    x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    os.chdir('..')
    np.savetxt("test_data.csv",x_test_2d,delimiter=",")
    np.savetxt("test_labels.csv",y_test,delimiter=",")


def make_train_data_orig():
    arr_features=[]
    cd = os.getcwd()
    logger.debug("curdir", cd)
    os.chdir('genres')
    #os.chdir('train')
    #genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    genres = 'blues'.split()
    for idx,distort in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(distort):
            #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
<<<<<<< Updated upstream
            y, sr = librosa.load(distort+'/'+fname, duration=10)
            print(distort+'/'+fname)
            print(y)
            print("size" + str(len(y)))
            print(sr)
=======
            y, sr = librosa.load(distort+'/'+fname, duration=30)
            logger.debug(distort+'/'+fname)
            logger.debug(y)
            logger.debug("size" + str(len(y)))
            logger.debug(sr)
>>>>>>> Stashed changes
            _features=extract_features_limited(y=y,sr=sr)
            arr_features.append(_features)
            arr_features.append(idx)

    df=pd.DataFrame(data=arr_features)
    logger.debug(df.head())
    logger.debug(df.shape)
    os.chdir('..')
    df.to_csv('train_data.csv',index=False)


def make_distortion_data_all():
    logger.info("Start Make Distortion Data All")
    os.chdir('genres')
    make_distortion_current_folder()
    os.chdir('test')
    make_distortion_current_folder()
    os.chdir('..')
    os.chdir('..')

def make_distortion_current_folder():
    logger.info("Generating Discontinuity")
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
    #os.chdir('train')
    genres = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for idx,distort in tqdm(enumerate(genres),total=len(genres)):
        
        if distort == 'randomsilence': #ToDo complete this list
<<<<<<< Updated upstream
            for fname in os.listdir(distort):
                
=======
            logger.info("Generating Random Silence 1s")
            for fname in os.listdir(distort):             
>>>>>>> Stashed changes
                generate_randomsilence(distort+'/'+fname)        
        if distort == 'click_n_pop':
            logger.info("Generating Click and Pops")
            for fname in os.listdir(distort):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_click_n_pops(distort+'/'+fname)
        if distort == 'discontinuity':
            logger.info("Generating Discontinuity")
            for fname in os.listdir(distort):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_discontinuity(distort+'/'+fname)
        if distort == 'hum':
            logger.info("Generating Hum")
            for fname in os.listdir(distort):

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
                generate_saturated(genre+'/'+fname) 
        if genre == 'rock':
            for fname in os.listdir(genre):
                generate_randomsilence5(genre+'/'+fname)
    
                generate_hum(distort+'/'+fname)
        if distort == 'white_noise2':
            logger.info("Generating White Noise #2")
            for fname in os.listdir(distort):
                generate_white_noise2(distort+'/'+fname)
        if distort == 'hum2':
            logger.info("Generating Hum #2")
            for fname in os.listdir(distort):
                generate_hum2(distort+'/'+fname) 
        if distort == 'oversampling':
            logger.info("Generating Oversampling")
            for fname in os.listdir(distort):
                generate_oversampling(distort+'/'+fname) 
        if distort == 'undersampling':
            logger.info("Generating Undersampling")
            for fname in os.listdir(distort):
                generate_undersampling(distort+'/'+fname) 
        if distort == 'saturated':
            logger.info("Generating Saturated")
            for fname in os.listdir(distort):
                generate_saturated(distort+'/'+fname) 
        if distort == 'randomsilence5':
            logger.info("Generating Random Silence 5s")
            for fname in os.listdir(distort):
                generate_randomsilence5(distort+'/'+fname)
    os.chdir('..')         

def make_test_data():
    logger.info("Make test data")
    arr_features=[]
    os.chdir('genres')
    #genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    distortions = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for fname in tqdm(os.listdir('test'),total=10*len(genres)):
            y, sr = librosa.load('test/'+fname, duration=30)
            dict_features=extract_features(y=y,sr=sr)
            dict_features['label']=genres.index(fname.split('.')[0])
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    logger.debug(df.head())
    logger.debug(df.shape)
    os.chdir('..')
    df.to_csv('test_data.csv',index=False)



if __name__=='__main__':
    #ToDo insert feature of introducing distortions
    #Blues = random silence
    # Classical = Clicks and pops
    # Country = Discontinuity
    # Disco = Hum introduction
    # hiphop  = Inter-sample peaks 
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('test_log.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    make_distortion_data_all()
    make_train_data_column()
    make_test_data_column()
    run_nn()