import os
from tqdm import tqdm

def rename_folder():
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    distortions = 'randomsilence click_n_pop discontinuity hum white_noise2 hum2 oversampling undersampling saturated randomsilence5'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        if genre == 'blues': 
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):   
                targ = target_main+'/'+fname
                dest = f_name=target_dest+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)
            os.rename(target_main, target_dest)
        if genre == 'classical':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest) 
            os.rename(target_main, target_dest)
        if genre == 'country':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)  
            os.rename(target_main, target_dest)
        if genre == 'disco':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)
            os.rename(target_main, target_dest)
        if genre == 'hiphop':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)
            os.rename(target_main, target_dest)
        if genre == 'jazz':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest) 
            os.rename(target_main, target_dest)
        if genre == 'metal':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)
            os.rename(target_main, target_dest)
        if genre == 'pop':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)
            os.rename(target_main, target_dest)
        if genre == 'reggae':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)
            os.rename(target_main, target_dest)
        if genre == 'rock':
            target_main = genre
            target_dest = distortions[idx]
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest) 
            os.rename(target_main, target_dest)
    os.chdir('..')

def function_rename_folder():
    rename_folder()
    os.chdir('genres/test')
    rename_folder()
    os.chdir('..')
    
if __name__=='__main__':
    #ToDo insert feature of introducing distortions
    #Blues = random silence
    # Classical = Clicks and pops
    # Country = Discontinuity
    # Disco = Hum introduction
    # hiphop  = Inter-sample peaks 
    function_rename_folder()