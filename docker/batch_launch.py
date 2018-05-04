from .launch_job import launch_job
import argparse


def launch_batch_job(njogs, *args)
for i in range(nj):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='write json configuration for ngc')
    parser.add_argument('run_script', type=str, help='relative path to the script to launch', default="")
    parser.add_argument('hyper', type=str, help='relative path to hyperparams file', default="")
    parser.add_argument('--int', default='False', type=str, help='interactive')
    parser.add_argument('--arg', default='', type=str, help='additional arguments')
    parser.add_argument('--name', default='', type=str, help='additional arguments')
    parser.add_argument('--ngpu', default=8, type=int, help='number of gpus')
    args = parser.parse_args()

    launch_batch_job(args.run_script, args.hyper, args.arg, args.interactiv, args.name, args.npgu)