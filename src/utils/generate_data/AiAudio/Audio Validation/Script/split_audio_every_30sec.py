#!/usr/bin/env python
#Program to segment audio to 30 s duration.


import os
import sys
from scipy.io import wavfile
import random as rd
import pyautogui as py


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

def import_audio():
    """Example list of commands."""
    do_command('Import2: Filename="C:/Users/mshahulh/OneDrive - Intel Corporation/Desktop/Basic_Media/Media/AVC4K/4K_Resolution_DP_Reel_(4480_x_1920).mp4"')
    
def get_length():
    """Import wavfile to convert to array form"""
    samplerate, data = wavfile.read('C:/Users/mshahulh/OneDrive - Intel Corporation/Desktop/test_export.wav')
    length = data.shape[0] / samplerate
    return length

def slice_audio():
    '''trying to slice audio in 30s interval'''
    do_command('RegularIntervalLabels:adjust="Yes" firstnum="27" interval="30" labeltext="Lab" mode="Interval" region="30" totalnum="100" verbose="None" zeros="TwoBefore"')
    
def generate_rand():
    '''Get Lenght first to generate random range'''
    length = get_length()
    print("this is length from add_noise function", length)
    """Add noise, distortion and silence on audio sample"""
    interval = 2
    rand_StartTime = abs(rd.uniform(0, length - 2))
    print("random start time",rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    do_command(crt_str)

def add_noise():
    do_command('Silence: UsePreset="<Current Settings>"')

def export_label():
    do_command('ExportMultiple: Folder="C:/Users/mshahulh/OneDrive - Intel Corporation/Desktop/AI Audio Validation Project/Input"')

        
def main():
    """Interactive command-line for PipeClient"""   
    
    #get_length()
    import_audio()
    #generate_rand()
    #add_noise()
    slice_audio()
    export_label()

main()
