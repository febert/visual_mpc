"this program runs on NGC and syncs data with a local master machine"
import time
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
import os
import tensorflow as tf
import ray



master = 'deepthought'

master_base_dir = '/home/ngc/Documents/visual_mpc/experiments/cem_exp/onpolicy'
master_modeldata_dir = master_base_dir + '/modeldata'
master_logging_dir = '/logging'
remote_datadir = '/raid/ngc/pushing_data/onpolicy/distributed_pushing/train'


@ray.remote
def sync(node_id, conf):
    logging_dir = conf['agent']['logging_dir']
    logger = Logger(logging_dir, 'sync_node{}.txt'.format(node_id))
    logger.log('started remote sync process on node{}'.format(node_id))

    # local means "locally" in the container on ngc
    local_modeldata_dir = conf['agent'][['result_dir']] + '/modeldata'
    local_datadir = conf['agent']['data_save_dir']

    if not os.path.exists(local_modeldata_dir):
        os.makedirs(local_modeldata_dir)

    while True:
        # get latest weights form master
        # rsync --ignore-existing deepthought:~/test .
        cmd = 'rsync --ignore-existing {}:{} {}'.format(master, master_modeldata_dir, local_modeldata_dir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)
        # consider --delete option

        # transfer tfrecords to master
        cmd = 'rsync --ignore-existing {} {}:{}'.format(local_datadir, master, remote_datadir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)

        # transfer logfiles to master
        os.system('rsync --ignore-existing {} {}:{}'.format(logging_dir, master, master_logging_dir))

        time.sleep(5)

# if __name__ == '__main__':
#     sync(node_id, conf)