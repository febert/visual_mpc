import os
import tensorflow as tf
import ray
master = 'deepthought'

@ray.remote
def sync(node_id, conf):

    modeldata_dir = conf['agent'][['result_dir']] + 'modeldata'
    if not os.path.exists(modeldata_dir):
        os.makedirs(modeldata_dir)

    while True:
        # get latest weights form master
        # rsync --ignore-existing deepthought:~/test .
        os.system('rsync --ignore-existing {}:{} {}'.format(master, remote_modeldata_dir, modeldata_dir))

        # transfer tfrecords to master
        os.system('rsync --ignore-existing')

        # transfer logfiles to master
        os.system('rsync --ignore-existing')

if __name__ == '__main__':
    sync(node_id)