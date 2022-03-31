import os
import sys
import random as rd
from tqdm import tqdm

if sys.platform == 'win32':
    print("pipe-test.py, running on windows")
    TONAME = '\\\\.\\pipe\\ToSrvPipe'
    FROMNAME = '\\\\.\\pipe\\FromSrvPipe'
    EOL = '\r\n\0'
else:
    print("pipe-test.py, running on linux or mac")
    TONAME = '/tmp/audacity_script_pipe.to.' + str(os.getuid())
    FROMNAME = '/tmp/audacity_script_pipe.from.' + str(os.getuid())
    EOL = '\n'

print("Write to  \"" + TONAME +"\"")
if not os.path.exists(TONAME):
    print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

print("Read from \"" + FROMNAME +"\"")
if not os.path.exists(FROMNAME):
    print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

print("-- Both pipes exist.  Good.")

TOFILE = open(TONAME, 'w')
print("-- File to write to has been opened")
FROMFILE = open(FROMNAME, 'rt')
print("-- File to read from has now been opened too\r\n")
cwd = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(cwd))
def send_command(command):
    """Send a single command."""
    print("Send: >>> \n"+command)
    TOFILE.write(command + EOL)
    TOFILE.flush()

def get_response():
    """Return the command response."""
    result = ''
    line = ''
    while True:
        result += line
        line = FROMFILE.readline()
        if line == '\n' and len(result) > 0:
            break
    return result

def do_command(command):
    """Send one command, and return the response."""
    send_command(command)
    response = get_response()
    print("Rcvd: <<< \n" + response)
    return response
    
def make_distortion_data():
    arr_features=[]
    os.chdir('genres')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        if genre == 'jazz': 
            for fname in os.listdir(genre):
                generate_inter_sample_peak("genres/"+genre+"/" +fname)        
        if genre == 'metal':
            for fname in os.listdir(genre):
                generate_distortion_expand_and_compress("genres/"+genre+'/'+fname)
        if genre == 'pop':
            for fname in os.listdir(genre):
                generate_distortion_fuzzbox("genres/"+genre+'/'+fname)
        if genre == 'reggae':
            for fname in os.listdir(genre):
                generate_distortion_hard_clipping("genres/"+genre+'/'+fname)
        if genre == 'rock':
            for fname in os.listdir(genre):
                generate_distortion_hard_overdrive("genres/"+genre+'/'+fname)

def generate_distortion_expand_and_compress(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    print("temp_str is,", temp_str)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str) 
    create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="30" Parameter_2="80" Repeats="0" Threshold_dB="-6" Type="Expand and Compress"'
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')

def generate_distortion_fuzzbox(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    print("temp_str is,", temp_str)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="80" Parameter_2="80" Repeats="0" Threshold_dB="-30" Type="Soft Clipping"'
    do_command(crt_str)
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')

def generate_distortion_hard_clipping(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    print("temp_str is,", temp_str)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str) 
 create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="0" Parameter_2="80" Repeats="0" Threshold_dB="-12" Type="Hard Clipping"'
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')

def generate_distortion_hard_overdrive(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    print("temp_str is,", temp_str)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str)
    create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="90" Parameter_2="80" Repeats="0" Threshold_dB="-6" Type="Hard Overdrive"'
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')
    
def generate_distortion_leveller_heaviest(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    print("temp_str is,", temp_str)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str)
    create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="0" Parameter_2="50" Repeats="5" Threshold_dB="-6" Type="Leveller"'
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')
    
def generate_distortion_soft_clipping(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}'.format(audio_file)
    print("temp_str is,", temp_str)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str)    
    create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="50" Parameter_2="80" Repeats="0" Threshold_dB="-12" Type="Soft Clipping"'
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')

def generate_distortion_hard_clipping(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str)
    create_distortion = 'Distortion: DC_Block="0" Noise_Floor="-70" Parameter_1="20" Parameter_2="80" Repeats="0" Threshold_dB="-6" Type="Medium Overdrive"'
    do_command(create_distortion)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')
    
def generate_inter_sample_peak(audio_file):
    length = 30
    temp_str = 'Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file)
    do_command(temp_str)
    interval = rd.randint(2, 5)
    rand_StartTime = abs(rd.uniform(0, length - interval))
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str)
    create_peak = 'Normalize: ApplyGain="1" PeakLevel="0" RemoveDcOffset="0" StereoIndependent="0"'
    do_command(create_peak)
    do_command('SelectAll:')
    temp_stri = 'Export2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Documents/Audacity/{}"'.format(audio_file) + ' NumChannels="1"'
    do_command(temp_stri)
    do_command('RemoveTracks:')

def main():
    """Interactive command-line for PipeClient"""
    make_distortion_data()    

main()
