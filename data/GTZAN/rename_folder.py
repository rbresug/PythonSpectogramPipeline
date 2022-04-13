#This scripts can be use to rename the file by appending the distortion type name in front of the audio name.
#eg: jazz.00000.wav > randomsilence.jazz.00000.wav

import os
from tqdm import tqdm

def rename_folder():
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()    
    distortions ='randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for idx,genre in tqdm(enumerate(distortions),total=len(distortions)):
        if genre == 'randomsilence': 
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'click_n_pop':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'discontinuity':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'hum':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'white_noise2':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'hum2':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'oversampling':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'undersampling':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'saturated':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
        if genre == 'randomsilence5':
            target_main = genre
            target_dest = distortions
            for count,fname in enumerate(os.listdir(genre)):  
                targ = target_main+'/'+fname
                dest = target_main+'/'+target_dest[idx]+'.'+fname
                os.rename(targ, dest)
    os.chdir('..')

        
if __name__=='__main__': 
    rename_folder()