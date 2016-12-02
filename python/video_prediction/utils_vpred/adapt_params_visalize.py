# DATA_DIR = '/home/frederik/Documents/pushing_data/test'
# DATA_DIR = '/home/frederik/Documents/pushing_data/finer_temporal_resolution_substep10/test'

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

def adapt_params_visualize(conf, model):

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
    conf['visualize'] = conf['output_dir'] + '/' + model
    conf['event_log_dir'] = '/tmp'
    conf['visual_file'] = conf['data_dir'] + '/traj_0_to_255.tfrecords'

    return conf