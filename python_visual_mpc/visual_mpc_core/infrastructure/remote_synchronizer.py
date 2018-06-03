"this program runs on ngc and syncs data with a local master machine"
import time
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
import os
import tensorflow as tf
import ray
import pdb

master = 'deepthought'


# @ray.remote
def sync(node_id, conf, printout=False):
    experiment_name =str.split(conf['current_dir'], '/')[-1]

    master_datadir = '/raid/ngc2/pushing_data/cartgripper/onpolicy/{}'.format(experiment_name)
    master_scoredir = '/raid/ngc2/pushing_data/cartgripper/onpolicy/{}/scores'.format(experiment_name)

    exp_subpath = conf['current_dir'].partition('onpolicy')[2]

    master_base_dir = '/home/ngc2/Documents/visual_mpc/experiments/cem_exp/onpolicy' + exp_subpath
    master_modeldata_dir = master_base_dir + '/modeldata'
    master_logging_dir = master_base_dir + '/logging_datacollectors'

    logging_dir = conf['agent']['logging_dir']
    logger = Logger(logging_dir, 'sync_node{}.txt'.format(node_id), printout=printout)
    logger.log('started remote sync process on node{}'.format(node_id))

    # local means "locally" in the container on ngc2
    local_modeldata_dir = '/result/modeldata'
    local_datadir = '/result/data'
    local_scoredir = '/result/data/scores'

    if not os.path.exists(local_modeldata_dir):
        os.makedirs(local_modeldata_dir)

    while True:
        logger.log('get latest weights from master')
        cmd = 'rsync -rltgoDv --delete-after {}:{} {}'.format(master, master_modeldata_dir + '/', local_modeldata_dir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)

        transfer_tfrecs(local_datadir, master_datadir, logger, 'train')
        transfer_tfrecs(local_datadir, master_datadir, logger, 'val')

        logger.log('transfer scorefiles to master')
        cmd = 'rsync -a --update {} {}:{}'.format(local_scoredir + '/', master, master_scoredir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)

        logger.log('transfer logfiles to master')
        cmd = 'rsync -a --update {} {}:{}'.format(logging_dir + '/', master, master_logging_dir)
        logger.log('executing: {}'.format(cmd))
        os.system(cmd)

        time.sleep(10)


def transfer_tfrecs(local_datadir, master_datadir, logger, mode):
    logger.log('transfer tfrecords to master')
    cmd = 'rsync -a --update {} {}:{}'.format(local_datadir + '/' + mode + '/', master, master_datadir + '/' + mode)
    logger.log('executing: {}'.format(cmd))
    os.system(cmd)


if __name__ == '__main__':

    conf = {}
    conf['current_dir'] = '/home/ngc2/Documents/visual_mpc/experiments/cem_exp/onpolicy/distributed_pushing'
    conf['agent'] = {}
    conf['agent']['result_dir'] = '/result'
    conf['agent']['logging_dir'] = '/result/logging_node0'
    sync(0, conf, printout=True)