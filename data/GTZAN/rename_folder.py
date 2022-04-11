import os
from tqdm import tqdm

def rename_folder():
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    distortions = 'gaps hums discontinuity disco hiphop jazz metal pop reggae rock'.split()
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
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)            
        if genre == 'country':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)            
        if genre == 'disco':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)           
        if genre == 'hiphop':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)           
        if genre == 'jazz':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)            
        if genre == 'metal':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)           
        if genre == 'pop':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)            
        if genre == 'reggae':
            for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)            
        if genre == 'rock':
             for count,fname in enumerate(os.listdir(genre)):
                targ = genre+'/'+fname
                dest = f_name=distortions[idx]+'_'+str(count)+'.wav'
                os.rename(targ, genre+'/'+dest)           
    os.chdir('..')

if __name__=='__main__':
    #ToDo insert feature of introducing distortions
    #Blues = random silence
    # Classical = Clicks and pops
    # Country = Discontinuity
    # Disco = Hum introduction
    # hiphop  = Inter-sample peaks 
    rename_folder()