# tar the trajectroies in a dataset

from gather_data import make_traj_name_list
import tarfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
conf = {}
conf['source_basedirs'] = [os.environ['VMPC_DATA_DIR'] +'/datacol_appflow/data/train']
conf['adim'] = 3
conf['sdim'] = 6
conf['ngroup'] = 1000

traj_list = make_traj_name_list(conf, shuffle=False)

for traj in traj_list:
    with tarfile.open(traj + "/traj.tar", "w") as tar:
        print 'taring ', traj
        tar.add(traj)

    with tarfile.open(traj + "/traj.tar") as tar:
        img_stream = tar.extractfile(traj[1:] + '/images/im7.png')
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)

        # im = cv2.imdecode(tar.extractfile(traj[1:] + '/images/im7.png'), 1)
        plt.imshow(im)
        plt.show()



