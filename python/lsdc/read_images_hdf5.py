import numpy as np
import h5py
import imp
import os

from matplotlib import pylab as plt

from lsdc import __file__ as gps_filepath
gps_filepath = os.path.abspath(gps_filepath)
gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
exp_dir = gps_dir + 'experiments/' + 'lsdc_exp' + '/'
hyperparams_file = exp_dir + 'hyperparams.py'


hyperparams = imp.load_source('hyperparams', hyperparams_file)



print hyperparams.agent['image_dir']
with h5py.File(hyperparams.agent['image_dir'],'r') as hf:
    print('List of items in the base directory:', hf.items())

    gp1 = hf.get('sample_no0')
    gp2 = hf.get('sample_no1')

    nparr = np.array(gp1)
    print nparr.shape

    im1 = plt.imshow(nparr[1,:,:,:])
    plt.show()


