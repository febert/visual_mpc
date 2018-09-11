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
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import npy_to_gif

class SaverWrapper:
    def __init__(self, save_dir, gen_T, mode):
        print('Saving to {} with mode: {}'.format(save_dir, mode))
        m = Manager()
        self._record_queue = m.Queue()
        if mode == 'train':
            split = (1., 0., 0.)
        elif mode == 'val':
            split = (0., 1., 0.)
        elif mode == 'test':
            split = (0., 0., 1.)
        else:
            raise NotImplementedError('Mode {} not implemented'.format(mode))
        worker_args = (self._record_queue, save_dir, gen_T, False, 16, 0, split)
        self._record_saver_proc = Process(target=record_worker, args=worker_args)
        self._record_saver_proc.start()

    def put(self, batch_dict):
        self._record_queue.put((batch_dict, None, None))

    def close(self):
        self._record_queue.put(None)
        self._record_saver_proc.join()


def last_non_zero(array):
    i = -1
    while -i <= len(array):
        if len(array[i]) > 0:
            return array[i]
        i -=1
    return ''


def create_saver(active_dataset, active_datapath, save_dir, mode):
    images, actions, state, T = active_dataset['images', mode], active_dataset['actions', mode], \
                                active_dataset['state', mode], active_dataset.T

    saver = SaverWrapper('{}/{}/'.format(save_dir, last_non_zero(active_datapath.split('/'))), 0, mode)
    return images, actions, state, T, saver


def generate_samples(netconf, data_paths, save_dir, noise_sigma, gripper_dim, gpu_id, ngpu, make_summary = False):
    gen_T, summaries = netconf['sequence_length'], []       # video predictions required generation to equal trained sequence_length

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

    for mode in BaseVideoDataset.MODES:
        active_dataset = 0
        one_hot_throwaway = None
        images, actions, state, T, saver = create_saver(datasets[active_dataset], data_paths[active_dataset], save_dir, mode)
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

                if gripper_dim:
                    n_steps = gen_T // 3
                    gen_noise = np.matmul(noise_sigma[:-1, :-1], np.random.normal(size=(netconf['batch_size'] - 1, n_steps,
                                                                              gen_actions.shape[-1] - 1, 1)))[:, :, :, 0]
                    gen_noise = np.repeat(gen_noise, 3, axis=1)[:, netconf['context_frames']:]
                    gripper_noise = [1. - noise_sigma[-1, -1], noise_sigma[-1, -1]]
                    gripper_noise = np.random.choice([1, -1], size=(netconf['batch_size'] - 1, n_steps),
                                                     p=gripper_noise).astype(np.float32)
                    gripper_noise = np.repeat(gripper_noise, 3, axis=1)[:, netconf['context_frames']:]
                    gen_actions[1:, netconf['context_frames']:, :-1] += gen_noise
                    gen_actions[1:, netconf['context_frames']:, -1] *= gripper_noise
                else:
                    gen_noise = np.matmul(noise_sigma, np.random.normal(size=(netconf['batch_size'] - 1, n_steps,
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
                if summaries is not None and make_summary and np.random.uniform() < 0.001:
                    tile = [batch_data['ground_truth_video'][:, 0]] + [batch_data['vidpred_real_actions'][:, 0, 0]]
                    num_samps = netconf['batch_size'] - 1
                    tile = tile + [batch_data['vidpred_random_actions'][:, i, 0] for i in range(num_samps)]
                    summaries.append(np.concatenate(tile, axis=2))
                    print('appended summary: {}'.format(len(summaries)))
                    if len(summaries) == 20:
                        summaries = np.concatenate(summaries, axis=1)
                        npy_to_gif([summaries[i] for i in range(summaries.shape[0])], './summary')
                        summaries = None

            except tf.errors.OutOfRangeError:
                active_dataset += 1
                saver.close()
                if active_dataset >= len(data_paths):
                    print('Generation Completed')
                    break
                images, actions, state, T, saver = create_saver(datasets[active_dataset], data_paths[active_dataset],
                                                                save_dir, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path', type=str, help="Path to the video predictor conf file")
    parser.add_argument('data_paths', type=str, help="Paths to video prediction datasets seperated by colons")
    parser.add_argument('--num_negative', type=int, default=5, help='number of negative samples to create per positive')
    parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--noise', nargs='+', type=float, default=[0.03, 0.03, 0.05, 8.])
    parser.add_argument('--gripper_dim', action="store_true", default=False)
    parser.add_argument('--summary_gif', action="store_true", default=False)

    args = parser.parse_args()

    loader = importlib.machinery.SourceFileLoader('mod_hyper', args.conf_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    netconf = mod.configuration
    netconf['batch_size'] = args.num_negative + 1

    splits = args.conf_path.split('/')
    if len(splits) == 1:
        save_dir = './'
    else:
        save_dir = '/'.join(splits[:-1])

    data_paths = args.data_paths.split(':')
    noise_vec = list(args.noise)

    if args.gripper_dim:
        assert 0 <= noise_vec[-1] <= 1, "noise_vec[-1] should be valid probability"
    generate_samples(netconf, data_paths, save_dir, np.diag(noise_vec), args.gripper_dim,
                     args.gpu_id, args.ngpu, args.summary_gif)
