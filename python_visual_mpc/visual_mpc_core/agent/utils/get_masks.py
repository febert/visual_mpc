import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2

def get_image(sim, agentparams):
    width = agentparams['viewer_image_width']
    height = agentparams['viewer_image_height']
    return sim.render(width, height, camera_name="maincam")[::-1, :, :]

def get_obj_masks(sim, agentparams, include_arm=False):
    sdim = agentparams['sdim']
    large_ob_masks = []
    ob_masks = []
    complete_img = get_image(sim, agentparams)
    armmask = None
    # plt.imshow(complete_img)
    # plt.show()
    if include_arm:
        qpos = copy.deepcopy(sim.data.qpos)
        qpos[2] -= 10
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim.set_state(sim_state)
        sim.forward()
        img = get_image(sim, agentparams)
        armmask = 1 - np.uint8(np.all(complete_img == img, axis=-1)) * 1
        qpos[2] += 10
        sim_state.qpos[:] = qpos
        sim.set_state(sim_state)
        sim.forward()

    for i in range(agentparams['num_objects']):
        qpos = copy.deepcopy(sim.data.qpos)
        qpos[sdim//2 + 2+ i*7] -= 1
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim.set_state(sim_state)
        sim.forward()
        img = get_image(sim, agentparams)
        # plt.imshow(img)
        # plt.show()
        mask = 1 - np.uint8(np.all(complete_img == img, axis=-1)) * 1
        qpos[sdim//2 + 2+ i * 7] += 1
        sim_state.qpos[:] = qpos
        sim.set_state(sim_state)
        sim.forward()

        large_ob_masks.append(mask)
        ob_masks.append(cv2.resize(mask, dsize=(
            agentparams['image_width'], agentparams['image_height']), interpolation=cv2.INTER_NEAREST))
        # plt.imshow(mask.squeeze())
        # plt.show()
    ob_masks = np.stack(ob_masks, 0)
    large_ob_masks = np.stack(large_ob_masks, 0)

    return ob_masks, large_ob_masks
