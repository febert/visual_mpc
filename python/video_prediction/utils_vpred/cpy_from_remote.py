import shutil
import argparse
import lsdc
import os
import pdb

def main():

    lsdc_basedir = lsdc.__file__

    TEN_DATA_LOC = '/'.join(str.split(lsdc_basedir, '/')[:-3]) + '/tensorflow_data'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str, help='name of folder in tensorflowdata')
    parser.add_argument('--remote', type=str, help='remote host name', default='newton2')

    args = parser.parse_args()
    exp_name = args.experiment
    remotemachine = args.remote

    if os.path.exists(TEN_DATA_LOC + '/' + exp_name):
        TEN_DATA_LOC = TEN_DATA_LOC + '/' + exp_name
    else:
        if os.path.exists(TEN_DATA_LOC + '/hidden_state/' + exp_name):
            print 'found in hidden_state folder'
            TEN_DATA_LOC = TEN_DATA_LOC + '/hidden_state/' + exp_name
        else:
            raise ValueError('folder not found')

    os.chdir(TEN_DATA_LOC)


    TEN_DATA_REMOTE = '/home/febert/' + '/'.join(str.split(TEN_DATA_LOC, '/')[3:]) + '/modeldata'

    print 'going to folder ', os.getcwd()
    cmd ="scp -r febert@{0}:{1} . ".format(remotemachine, TEN_DATA_REMOTE)
    print 'executing ', cmd
    os.system(cmd)


if __name__ == '__main__':
    main()



