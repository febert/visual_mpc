import cPickle

basedir = '/media/febert/harddisk/febert/sawyer_data/newrecording'

gr = 0

trstart = 0

trend = 10

for tr in range(trstart, trend, 2):

    file = basedir + '/traj_group{}/traj{}/joint_angles_traj{}.pkl'.format(gr, tr,tr)
    dict  =cPickle.load(open(file, "rb"))

    actions = dict['actions']

    joint_angles = dict['jointangles']


    endeffector_pos = dict['endeffector_pos']
    print '###########################################'
    print 'file:', file

    print '------------------------'
    print 'actions:'
    print 'shape:', actions.shape
    print actions

    print '------------------------'
    print 'endeffector_pos:'
    print 'shape:', endeffector_pos.shape
    print endeffector_pos

    print '------------------------'
    print 'joint_angles:'
    print 'shape:', joint_angles.shape
    print joint_angles


