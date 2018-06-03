from multiprocessing import Pool
import sys
import argparse
import os
import importlib.machinery
import importlib.util

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.online_training.replay_buffer import ReplayBuffer_Loadfiles
from python_visual_mpc.video_prediction.online_training.trainvid_online import trainvid_online
from python_visual_mpc.video_prediction.online_training.trainvid_online_alexmodel import trainvid_online_alexmodel
import matplotlib; matplotlib.use('Agg'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model

def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
    parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
    parser.add_argument('--nsplit', type=int, help='number of splits', default=-1)
    parser.add_argument('--isplit', type=int, help='split id', default=-1)
    parser.add_argument('--iex', type=int, help='if different from -1 use only do example', default=-1)
    parser.add_argument('--printout', type=int, help='print to console if 1', default=0)


    args = parser.parse_args()
    trainvid_conf_file = args.experiment
    conf = load_module(trainvid_conf_file, 'trainvid_conf')

    if 'RESULT_DIR' in os.environ:
        conf['result_dir'] = os.environ['RESULT_DIR']
    else:
        conf['result_dir'] = conf['current_dir']

    printout = bool(args.printout)
    gpu_id = args.gpu_id

    logging_dir = conf['current_dir'] + '/logging'
    conf['logging_dir'] = logging_dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    train_rb = ReplayBuffer_Loadfiles(conf, mode='train', printout=printout)
    val_rb = ReplayBuffer_Loadfiles(conf, mode='val', printout=printout)

    if conf['pred_model'] == Alex_Interface_Model:
        trainvid_online_alexmodel(train_rb, val_rb, conf, logging_dir, gpu_id, printout=True)
    else:
        trainvid_online(train_rb, val_rb, conf, logging_dir, gpu_id, printout=True)

def load_module(hyperparams_file, name):
    loader = importlib.machinery.SourceFileLoader(name, hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    hyperparams = mod.config
    return hyperparams

if __name__ == '__main__':
    main()