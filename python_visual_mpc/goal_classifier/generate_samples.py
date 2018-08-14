import importlib.machinery
import importlib.util
import argparse
from python_visual_mpc.visual_mpc_core.Datasets.base_dataset import BaseVideoDataset
import tensorflow as tf
import os
import numpy as np
from python_visual_mpc.visual_mpc_core.agent.utils.traj_saver import record_worker
from multiprocessing import Process, Manager
from collections import OrderedDict


class SaverWrapper:
    def __init__(self, save_dir, gen_T):
        print('Saving to {}'.format(save_dir))
        m = Manager()
        self._record_queue = m.Queue()
        self._record_saver_proc = Process(target=record_worker,args=(self._record_queue, save_dir, gen_T, False, 16, 0))
        self._record_saver_proc.start()

    def put(self, batch_dict):
        self._record_queue.put((batch_dict, None, None))

    def close(self):
        self._record_queue.put(None)
        self._record_saver_proc.join()


def first_non_zero(array):
    i = -1
    while -i <= len(array):
        if len(array[i]) > 0:
            return array[i]
        i -=1
    return ''


def generate_samples(netconf, data_paths, save_dir, noise_sigma, gpu_id, ngpu):
    gen_T = netconf['sequence_length']       # video predictions required generation to equal trained sequence_length

    datasets = [BaseVideoDataset(data_dir, 1, hparams_dict={'num_epochs':1}) for data_dir in data_paths]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)
    data_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(data_sess)
    data_sess.run(tf.global_variables_initializer())

    predictor = netconf['setup_predictor']({},  netconf, gpu_id, ngpu)

    pad_frames = netconf['context_frames'] + gen_T
    assert all([dataset.T - pad_frames >= 0 for dataset in datasets]), \
        "All datasets should have at least {} images".format(pad_frames)
    active_dataset = 0
    one_hot_throwaway = None
    images, actions, state, T = datasets[active_dataset]['images'], datasets[active_dataset]['actions'], \
                                datasets[active_dataset]['state'], datasets[active_dataset].T
    saver = SaverWrapper('{}/{}/'.format(save_dir, first_non_zero(data_paths[active_dataset].split('/'))), 0)

    while True:
        try:
            samp_images, samp_actions, samp_states = data_sess.run([images, actions, state])
            start = np.random.randint(T - pad_frames + 1)
            after_context = start + netconf['context_frames']
            context_images = samp_images[:, start:after_context].astype(np.float32) / 255.
            if one_hot_throwaway is None:    # needed because some video prediction models can't handle distrib = None case
                img_shape = list(context_images.shape)
                img_shape[-1] = 2
                one_hot_throwaway = np.ones(tuple(img_shape), dtype=np.float32)
                one_hot_throwaway /= np.prod(img_shape)

            context_states = samp_states[:, start:after_context]
            gen_actions = np.repeat(samp_actions[:, start:start+gen_T], netconf['batch_size'], axis=0)
            n_steps = gen_T - netconf['context_frames']
            gen_noise = np.matmul(noise_sigma, np.random.normal(size=(netconf['batch_size']-1, n_steps,
                                                                      gen_actions.shape[-1], 1)))
            gen_actions[1:, netconf['context_frames']:] += gen_noise.reshape((netconf['batch_size'] - 1, n_steps, -1))

            gen_images = predictor(input_images=context_images, input_state=context_states,
                                   input_actions=gen_actions, input_one_hot_images=one_hot_throwaway)[0]

            gen_images = np.swapaxes((gen_images * 255.).astype(np.uint8), 0, 1)
            gen_actions = np.swapaxes(gen_actions, 0, 1)

            batch_data = OrderedDict()
            batch_data['ground_truth_video'] = samp_images[0, after_context:start + gen_T]
            batch_data['real_actions'] = np.expand_dims(gen_actions[netconf['context_frames']:, 0], 1)
            batch_data['vidpred_real_actions'] = np.expand_dims(gen_images[:, 0], 1)
            batch_data['random_actions'] = gen_actions[netconf['context_frames']:, 1:]
            batch_data['vidpred_random_actions'] = gen_images[:, 1:]
            saver.put(batch_data)
        except tf.errors.OutOfRangeError:
            active_dataset += 1
            if active_dataset >= len(data_paths):
                print('Generation Completed')
                break
            images, actions, state, T = datasets[active_dataset]['images'], datasets[active_dataset]['actions'], \
                                      datasets[active_dataset]['state'], datasets[active_dataset].T
            saver = SaverWrapper('{}/{}/'.format(save_dir, first_non_zero(data_paths[active_dataset].split('/'))), 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path', type=str, help="Path to the video predictor conf file")
    parser.add_argument('data_paths', type=str, help="Paths to video prediction datasets seperated by colons")
    parser.add_argument('--num_negative', type=int, default=5, help='number of negative samples to create per positive')
    parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--noise', nargs='+', type=float, default=[0.03, 0.03, 0.05, 8.])
    args = parser.parse_args()

    loader = importlib.machinery.SourceFileLoader('mod_hyper', args.conf_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    netconf = mod.configuration
    netconf['batch_size'] = args.num_negative

    splits = args.conf_path.split('/')
    if len(splits) == 1:
        save_dir = './'
    else:
        save_dir = '/'.join(splits[:-1])

    data_paths = args.data_paths.split(':')
    noise_vec = list(args.noise)
    noise_vec[3] = np.pi / noise_vec[3]

    generate_samples(netconf, data_paths, save_dir, np.diag(noise_vec), args.gpu_id, args.ngpu)