import os
import sys
import random as rd

path = "C:/Users/mshahulh/OneDrive - Intel Corporation/Desktop/AI Audio Validation Project/Input"
path_out = "C:/Users/mshahulh/OneDrive - Intel Corporation/Desktop/AI Audio Validation Project/Output_Robot"

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

def generate_rand():
    '''Get Lenght first to generate random range'''
    print("generate_randOm")
    length = 30
    interval = 5
    rand_StartTime = abs(rd.uniform(0, length - interval))
    print("rand startime", rand_StartTime)
    crt_str ='Select: End="{sum}"'.format(sum=rand_StartTime + interval) + ' Mode="Set"' + ' Start="{rnd}"'.format(rnd = rand_StartTime)
    print("crt_str",crt_str)
    do_command(crt_str)
    rand_pitch = rd.uniform(-50, -85)
    pitch_str = 'ChangePitch: Percent="{ew}"'.format(ew = rand_pitch) + ' SBMS="1"'
    do_command(pitch_str)

def main():
    x = os.listdir(path)
    print(len(x))       
    for filename in (x):
        temp_str = 'Import2: Filename="{y}'.format(y = path) + '/{file_name}"'.format(file_name = filename)
        do_command(temp_str)
        generate_rand()
        do_command('SelectAll:')
        temp_stri = 'Export2: Filename="{z}'.format(z = path_out)  + '/{filenname}"'.format(filenname = filename) + ' NumChannels="2"'
        do_command(temp_stri)
        do_command('RemoveTracks:')
main()