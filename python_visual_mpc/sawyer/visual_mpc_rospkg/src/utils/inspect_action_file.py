import pickle

# basedir = '/media/febert/harddisk/febert/sawyer_data/newrecording'
basedir = '/storage/febert/sawyer_data/online_data'
gr = 1

trstart = 0

trend = 10

# for tr in range(trstart, trend, 2):
tr = 1449

file = basedir + '/traj_group{}/traj{}/joint_angles_traj{}.pkl'.format(gr, tr,tr)
dict  =pickle.load(open(file, "rb"))

actions = dict['actions']

joint_angles = dict['jointangles']


endeffector_pos = dict['endeffector_pos']
print('###########################################')
print('file:', file)

print('------------------------')
print('actions:')
print('shape:', actions.shape)
print(actions)

print('------------------------')
print('endeffector_pos:')
print('shape:', endeffector_pos.shape)
print(endeffector_pos)

print('------------------------')
print('joint_angles:')
print('shape:', joint_angles.shape)
print(joint_angles)

if 'track_desig' in dict:
    track_desig = dict['track_desig']
    print('------------------------')
    print('track_desig:')
    print('shape:', track_desig.shape)
    print(track_desig)

    goal_pos = dict['goal_pos']
    print('------------------------')
    print('goal_pos:')
    print('shape:', goal_pos.shape)
    print(goal_pos)



