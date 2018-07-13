import sys
sys.path.append('../')

from docker.launch_job import launch_job_func
import argparse

def launch_batch_job(njobs, *args):
    for i in range(njobs):
        launch_job_func(*args, nsplit=njobs, isplit=i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='write json configuration for ngc')
    parser.add_argument('run_script', type=str, help='relative path to the script to launch', default="")
    parser.add_argument('hyper', type=str, help='relative path to hyperparams file', default="")
    parser.add_argument('--int', default='False', type=str, help='interactive')
    parser.add_argument('--arg', default=' --nworkers 8', type=str, help='additional arguments')
    parser.add_argument('--name', default='', type=str, help='additional arguments')
    parser.add_argument('--ngpu', default=8, type=int, help='number of gpus per node')

    parser.add_argument('--nsplit', default=-1, type=int, help='number of splits')

    parser.add_argument('--test', default=0, type=int, help='testrun')
    args = parser.parse_args()

    launch_batch_job(args.nsplit, args.run_script, args.hyper, args.arg, args.int, args.name, args.ngpu, args.test)