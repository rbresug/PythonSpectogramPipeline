import os
import sys
import random as rd
import time
from pathlib import Path, PureWindowsPath

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

time.sleep(1)
TOFILE = open(TONAME, 'wt+')
print("-- File to write to has been opened")
time.sleep(1)
FROMFILE = open(FROMNAME, 'rt')
print("-- File to read from has now been opened too\r\n")

path = "c:/tempo/06_Work/InProgess/AI_work/Audio Automation/repo/clean/PythonSpectogramPipeline/src/utils/generate_data/AiAudio/Audio Validation/Audio/Input Audio"

path_out = "c:/tempo/06_Work/InProgess/AI_work/Audio Automation/repo/clean/PythonSpectogramPipeline/src/utils/generate_data/AiAudio/Audio Validation/Audio/Output Silence"

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
   
def generate_rand():
    '''Get Lenght first to generate random range'''
    print("generate_randOm")
    length = 30
    interval = 0.65
    rand_rand = abs(rd.randint(2, 10))
    output = [] 
    print("rand_rand is", rand_rand)
    print(range(rand_rand))
    for y in range(rand_rand):
        rand_StartTime = abs(rd.uniform(0, length - interval))
        output.append(rand_StartTime)
        crt_str ='Select: End="{sum}"'.format(sum=output[y] + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = output[y])
        print("crt_str",crt_str)
        do_command(crt_str)
        do_command('Silence: UsePreset="<Current Settings>"')

def temp_loop():
    filename = main()
    print("masuk temp_loop")
    
    #print(filename)

def main():
    """Interactive command-line for PipeClient"""
    x = os.listdir(path)
    print(len(x))       
    for y in range(31):
        temp_str = 'Import2: Filename="c:/tempo/06_Work/InProgess/AI_work/Audio Automation/repo/clean/PythonSpectogramPipeline/src/utils/generate_data/AiAudio/Audio Validation/Audio/Input Audio/30sec.wav"'
        print("temp_str is,", temp_str)
        do_command(temp_str)
        generate_rand()
        do_command('SelectAll:')
        temp_stri = 'Export2: Filename="c:/tempo/06_Work/InProgess/AI_work/Audio Automation/repo/clean/PythonSpectogramPipeline/src/utils/generate_data/AiAudio/Audio Validation/Audio/Output Silence/Audio{}.wav"'.format(y)
        do_command(temp_stri)
        do_command('RemoveTracks:')

main()
