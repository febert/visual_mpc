# creates a collection of random configurations for pushing
import numpy as np

import cPickle

alpha = np.random.uniform(0,360.0)
ob_pos = np.random.uniform(-0.4,0.4,2)
goalpoint = np.random.uniform(-0.4,0.4,2)

nconf = 20

traj = 0
n_reseed = 3
n_conf = -1

goalpoints = []
initialposes = []

for n in range(nconf):

    alpha = np.random.uniform(0, 360.0)
    ob_pos = np.random.uniform(-0.35, 0.35, 2)
    goalpoint = np.random.uniform(-0.4, 0.4, 2)

    n_conf += 1

    initpos = np.array([0., 0., 0., 0., ob_pos[0], ob_pos[1], 0., np.cos(alpha/2), 0, 0, np.sin(alpha/2)  #object pose (x,y,z, quat)
    ])

    goalpoints.append(goalpoint)
    initialposes.append(initpos)
    print 'initpose:', initpos
    print 'goalpoint', goalpoint

dict = {}

dict['goalpoints'] = goalpoints
dict['initialpos'] = initialposes

cPickle.dump(dict, open('benchmarkconfigs', mode='wb'))