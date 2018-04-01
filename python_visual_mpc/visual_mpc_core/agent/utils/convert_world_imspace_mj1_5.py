import numpy as np
import pickle

from python_visual_mpc import __file__ as python_vmpc_path
root_dir = '/'.join(str.split(python_vmpc_path, '/')[:-1])
mats = pickle.load(open(root_dir + '/visual_mpc_core/agent/utils/proj_mats.pkl', 'rb'), encoding='latin1')

VIEW = mats['viewport']
GL_MODELVIEW_MATRIX = mats['modelview']
GL_PROJECTION_MATRIX = mats['projection']

def project_point(p, return_zval=False):
    """
    projects a point from the world coordinate system to the screen coordinate system
    """
    p = p.astype(np.float32)
    # print(("p", p))
    # print("model view")
    # print(GL_MODELVIEW_MATRIX)
    # print("projection")
    # print(GL_PROJECTION_MATRIX)

    Vx, Vy, Vz = VIEW[2], VIEW[3], 1
    x0, y0 = VIEW[0], VIEW[1]

    obj_coord = np.concatenate([np.squeeze(p), np.array([1.])])
    eye_coord = np.dot(GL_MODELVIEW_MATRIX.T, obj_coord)
    clip_coord = np.dot(GL_PROJECTION_MATRIX.T, eye_coord)
    # print(("clip coord", clip_coord))

    ndc = clip_coord[:3] / clip_coord[3]  # everything should now be in -1 to 1!!
    col, row, z = (ndc[0]+1)*Vx/2 + x0, (-ndc[1]+1)*Vy/2 + y0, (ndc[2]+1)*Vz/2

    if return_zval:
        return row, col, z
    else:
        return row, col

def get_3D(r, c, z):
    r = float(r)
    c = float(c)
    z = float(z)

    Vx, Vy, Vz = VIEW[2], VIEW[3], 1
    x0, y0 = VIEW[0], VIEW[1]
    clip_x, clip_y, clip_z = 2 * (c - x0) / Vx - 1, -2 * (r - y0) / Vy + 1, 2 * z / Vz - 1
    PVM = np.dot(GL_PROJECTION_MATRIX.T, GL_MODELVIEW_MATRIX.T)
    pos = np.array([clip_x, clip_y, clip_z, 1])
    world_pos = np.dot(np.linalg.inv(PVM), pos)
    return world_pos[:3] / world_pos[3]
