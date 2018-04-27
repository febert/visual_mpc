from pyquaternion import Quaternion
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single
from python_visual_mpc.visual_mpc_core.agent.utils.convert_world_imspace_mj1_5 import project_point, get_3D
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import copy



def gen_gtruthdesig(curr_obj_pose, goal_obj_pose, curr_obj_mask, curr_dimage, npoints, agentparams, curr_image=None, goal_image=None):
    
    """
    generate pairs of designated pixels and goal pixels by sampling designated pixels on the current object mask and finding the corresponding goal pixels
    """
    if len(goal_obj_pose.shape) == 2:
        goal_obj_pose = goal_obj_pose[0]
    goal_pix = []
    desig_pix = []
    goal_pos = goal_obj_pose[:3]
    goal_quat = Quaternion(goal_obj_pose[3:])
    curr_pos = curr_obj_pose[:3]
    curr_quat = Quaternion(curr_obj_pose[3:])
    onobject = np.stack(np.where(np.squeeze(curr_obj_mask) != 0), 1)
    if len(onobject) == 0:
        print("zero pixel of object visible!")
        desig_pix = np.zeros((npoints, 2))
        goal_pix = np.zeros((npoints, 2))
        return desig_pix, goal_pix
    diff_quat = curr_quat.conjugate * goal_quat  # rotates vector form curr_quat to goal_quat
    dsample_factor = agentparams['viewer_image_height']/float(agentparams['image_height'])

    for i in range(npoints):
        id = np.random.choice(list(range(onobject.shape[0])))
        coord = onobject[id]
        desig_pix.append((coord/dsample_factor).astype(np.int))
        abs_pos_curr_sys = get_3D(coord[0], coord[1], curr_dimage[coord[0], coord[1]])
        rel_pos_curr_sys = abs_pos_curr_sys - curr_pos
        rel_pos_curr_sys = Quaternion(scalar=.0, vector=rel_pos_curr_sys)
        rel_pos_prev_sys = diff_quat * rel_pos_curr_sys * diff_quat.conjugate
        abs_pos_prev_sys = goal_pos + rel_pos_prev_sys.elements[1:]
        goal_prev_sys_imspace = np.array(project_point(abs_pos_prev_sys))
        goal_pix.append((goal_prev_sys_imspace/dsample_factor).astype(np.int))

        # plt.figure()
        # ax1 = plt.subplot(121)
        # ax2 = plt.subplot(122)
        # ax1.imshow(add_crosshairs_single(copy.deepcopy(curr_image), desig_pix[-1]))
        # ax2.imshow(add_crosshairs_single(copy.deepcopy(goal_image), goal_pix[-1]))
        # plt.show()
    return np.stack(desig_pix, axis=0), np.stack(goal_pix, axis=0)
