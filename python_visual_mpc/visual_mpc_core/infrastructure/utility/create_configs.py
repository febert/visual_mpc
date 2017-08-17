# creates a collection of random configurations for pushing
import numpy as np
import random
import cPickle


def create(filename, nconf):

    idx = -1

    goalpoints = []
    initialposes = []

    for n in range(nconf):

        alpha = np.random.uniform(0, 360.0)
        ob_pos = np.random.uniform(-0.35, 0.35, 2)
        goalpoint = np.random.uniform(-0.4, 0.4, 2)

        idx += 1

        initpos = np.array([0., 0., 0., 0., ob_pos[0], ob_pos[1], 0., np.cos(alpha/2), 0, 0, np.sin(alpha/2)  #object pose (x,y,z, quat)
        ])

        goalpoints.append(goalpoint)
        initialposes.append(initpos)
        # print 'config no: ', n
        # print 'initpose:', initpos
        # print 'goalpoint', goalpoint

    dict = {}

    dict['goalpoints'] = goalpoints
    dict['initialpos'] = initialposes

    cPickle.dump(dict, open(filename, mode='wb'))

    print "done."

def read(filename):
    confs = cPickle.load(open(filename, "rb"))
    goalpoints = confs['goalpoints']
    initialposes = confs['initialpos']

    for i in range(len(goalpoints)):
        print 'config {0}: goalpoint: {1}, initial object pos: {2}'.format(i, goalpoints[i], initialposes[i])


def create_easy_goalconfigs(targetfile = None):
    confs = cPickle.load(open('benchmarkconfigs_1e4', "rb"))

    initialposes = confs['initialpos']
    goalpoints = []

    for initpos in initialposes:

        outside = True

        while outside:
            shiftvec = np.random.uniform(-0.2, 0.2, 2)
            startpos = initpos[4:6]
            goalpos = startpos  + shiftvec
            if  goalpos[0] < .4  and \
                goalpos[0] > -.4 and \
                goalpos[1] < .4 and \
                goalpos[1] > -.4:
                outside = False

        goalpoints.append(goalpos)
        print 'original pos {}, new goalpos {}'.format(startpos, goalpos)

    dict = {}
    dict['goalpoints'] = goalpoints
    dict['initialpos'] = initialposes

    cPickle.dump(dict, open(targetfile, mode='wb'))


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print 'using seed', seed

    filename = 'benchmarkconfigs_1e4'
    nconf = 10000
    create(filename, nconf)

    # create_easy_goalconfigs(targetfile='configs_easy_goal_1e4')
    # read(filename)