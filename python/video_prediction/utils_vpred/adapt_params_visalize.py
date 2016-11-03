DATA_DIR = '/home/frederik/Documents/pushing_data/test'

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

def adapt_params_visualize(conf, model):

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data.' ,
    conf['visualize'] = conf['output_dir'] + '/' + model
    conf['event_log_dir'] = '/tmp'
    conf['visual_file'] = DATA_DIR + '/traj_66560_to_66815.tfrecords'

    return conf