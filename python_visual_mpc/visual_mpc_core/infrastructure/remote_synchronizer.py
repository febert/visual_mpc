"this program runs on NGC and syncs data with a local master machine"
import time
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
import os
import tensorflow as tf
import ray
import pdb

master = 'deepthought'

master_datadir = '/raid/ngc/pushing_data/onpolicy/distributed_pushing/train'

@ray.remote
def sync(node_id, conf, printout=False):
    exp_subpath = conf['current_dir'].partition('onpolicy')[2]

    master_base_dir = '/home/ngc/Documents/visual_mpc/experiments/cem_exp/onpolicy' + exp_subpath
    master_modeldata_dir = master_base_dir + '/modeldata'
    master_logging_dir = master_base_dir + '/logging_datacollectors'

    logging_dir = conf['agent']['logging_dir']
    logger = Logger(logging_dir, 'sync_node{}.txt'.format(node_id), printout=printout)
    logger.log('started remote sync process on node{}'.format(node_id))

    # local means "locally" in the container on ngc
    local_modeldata_dir = '/result/modeldata'
    local_datadir = '/result/data/train'

    if not os.path.exists(local_modeldata_dir):
        os.makedirs(local_modeldata_dir)

    while True:
        logger.log('get latest weights from master')
        # rsync --ignore-existing deepthought:~/test .
        # cmd = 'rsync -a {}:{} {}'.format(master, master_modeldata_dir + '/', local_modeldata_dir)
        cmd = 'rsync -rltgoDv --delete {}:{} {}'.format(master, master_modeldata_dir + '/', local_modeldata_dir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)
        # consider --delete option

        logger.log('transfer tfrecords to master')
        cmd = 'rsync -a --ignore-existing {} {}:{}'.format(local_datadir + '/', master, master_datadir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)

        logger.log('transfer logfiles to master')
        cmd = 'rsync -a --ignore-existing {} {}:{}'.format(logging_dir + '/', master, master_logging_dir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)

        time.sleep(10)

if __name__ == '__main__':

    conf = {}
    conf['current_dir'] = '/home/ngc/Documents/visual_mpc/experiments/cem_exp/onpolicy/distributed_pushing'
    conf['agent'] = {}
    conf['agent']['result_dir'] = '/result'
    conf['agent']['logging_dir'] = '/result/logging_node0'
    sync(0, conf, printout=True)