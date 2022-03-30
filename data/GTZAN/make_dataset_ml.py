import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew
from IPython.display import Audio

from essentia.standard import MonoLoader, MonoWriter, DiscontinuityDetector, FrameGenerator
from essentia import array
import matplotlib.pyplot as plt
import numpy as np
import random as rd


def generate_discontinuity(audio_file):

    plt.rcParams["figure.figsize"] = (12, 9)

    sr = 22050

    audio = MonoLoader(filename=audio_file, sampleRate =sr)()

    jump_starts = np.array([len(audio) // 4, len(audio) // 2])
    ground_truth = jump_starts / sr

    for start in jump_starts:
        l_amp = audio[start]
        # Remove samples until the jump produces a prominent discontinuity so it can be perceived.
        end = next(
            idx for idx, r_amp in enumerate(audio[start:], start) if abs(r_amp - l_amp) > 0.3
        )
        audio = np.delete(audio, range(start, end))
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

def generate_randomsilence(audio_file):
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    sr = 22050
    
    time_axis = np.arange(len(audio)) / sr
    gap_position = rd.randint(2,28)
    gap_duration = 0.5
    gap_start = gap_position * sr
    gap_end = int(gap_start + gap_duration * sr)
    print(gap_duration * sr)
    
    audio[gap_start: gap_end] = np.zeros(int(gap_duration * sr))
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

#25/3/22 mshahulh
def generate_click_n_pops(audio_file):
    sr = 22050

    audio = MonoLoader(filename=audio_file, sampleRate=sr)()
    
    # I purposely generate random int between 2 - 5 to varies the location of the click n pops in the audio
    rand_length = []
    for x in range(3):
        y = rd.randint(2,5)
        rand_length.append(y)
        
    click_locs = [
        len(audio) // rand_length[0],
        len(audio) // rand_length[1],
        len(audio) * 3 // rand_length[3]
                 ]
    for click_loc, amp in zip(click_locs, [.5, .15, .05]):
        audio[click_loc] += amp
    
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio)

#25/3/22 mshahulh
def generate_hum(audio_file):
    
    sr = 22050
    
    audio = MonoLoader(filename=audio_file, sampleRate=sr)()

    # Generate a 50Hz tone.
    time = np.arange(len(audio)) / sr
    freq = 50
    hum = np.sin(2 * np.pi * freq * time).astype(np.float32)

    # Add odd harmonics via clipping.
    hum = Clipper(min=-0.5, max=0.5)(hum)

    # Attenuate the hum 30 dB.
    hum *= db2amp(-30)

    audio_with_hum = audio + hum
    
    MonoWriter(filename=audio_file, sampleRate=sr, format='wav')(audio_with_hum)
#28/3/2022 mshahulh
def generate_white_noise(audio_file):
    
    sr = 22050
    
    audio = MonoLoader(filename= audio_file, sampleRate= sr)()
    audio *= db2amp(audio_db)
    
    noise_only = 1

    audio_db = -10
    noise_db = -60

    time = np.arange(len(time))
    
    noise = np.random.randn(len(time)).astype(np.float32)
    noise *= db2amp(noise_db)

    audio_power = instantPower(audio)
    noise_power = instantPower(noise)
    true_snr = 10 * np.log10( audio_power / noise_power)

    print('audio level: {:.2f}dB'.format(10. * np.log10(audio_power)))
    print('Noise level: {:.2f}dB'.format(10. * np.log10(noise_power)))
    print('SNR: {:.2f}dB'.format(true_snr))

    MonoWriter(filename = audio_file ,format='wav',sampleRate=sr)(audio + noise)

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
            dict_features=extract_features(y=y,sr=sr)
            dict_features['label']=idx
            arr_features.append(dict_features)

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
                generate_discontinuity(genre+'/'+fname)
        if genre == 'disco':
            for fname in os.listdir(genre):

                #23/03/22 RBresug: selected only 10 seconds because of error ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
                generate_hum(genre+'/'+fname)
        if genre == 'hiphop':
            for fname in os.listdir(genre):
                
                generate_white_noise(genre+'/'+fname)

def make_test_data():
    arr_features=[]
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
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

#30/03/22 ahmadbad:
#define the source audio and destination location  
def split_data():
    os.chdir('genres\\genres') #local path for audio file (source audio)
    dirName = r"C:\Ai_audio\genres\genres\test\\" #local path for test file (destination audio)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    # genres = 'blues'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        for i,fname in enumerate(os.listdir(genre)):
            audioname = (genre + "\\" + fname)   
            if i <= 10:
                print("inside loop: " + audioname)
                try:
                    # Create target Directory
                    os.mkdir(dirName)
                    print("Directory " , dirName ,  " Created ") 
                except FileExistsError:
                    print("Directory " , dirName ,  " already exists")
                shutil.move(audioname, dirName+fname)
            else:
                print("outside loop: " +audioname)

if __name__=='__main__':
    #ToDo insert feature of introducing distortions
    #Blues = random silence
    # Classical = Clicks and pops
    # Country = Discontinuity
    # Disco = Hum introduction
    # hiphop  = Inter-sample peaks 
    #make_distortion_data()
    #make_train_data()
    make_test_data()
    #split_data
