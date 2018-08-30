from mujoco_py import load_model_from_path, MjSim
import ray
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import npy_to_gif

@ray.remote(num_gpus=1)
class SimWorker(object):
    def __init__(self):
        print('created worker')

    def create_sim(self):
        self.sim = MjSim(load_model_from_path('/mount/visual_mpc/mjc_models/cartgripper_noautogen.xml'))
        # self.sim = MjSim(load_model_from_path('/mnt/sda1/visual_mpc/mjc_models/cartgripper_noautogen.xml'))

    def run_sim(self):
        print('startsim')
        images = []
        for i in range(5):
            self.sim.step()
            images.append(self.sim.render(100, 100))
        return images


def run():

    use_ray = True

    if use_ray:
        # ray.init(driver_mode=ray.PYTHON_MODE)
        ray.init()
        workers = []
        nworkers = 1
        for i in range(nworkers):
            workers.append(SimWorker.remote())

        id_list = []
        for i, worker in enumerate(workers):
            id_list.append(worker.create_sim.remote())
        res = [ray.get(id) for id in id_list]

        id_list = []
        for i, worker in enumerate(workers):
            id_list.append(worker.run_sim.remote())

        for id in id_list:
            images = ray.get(id)
            # npy_to_gif(images, '~/Desktop/video{}'.format(id))
            npy_to_gif(images, '/Desktop/video')

    else:
        worker = SimWorker()
        worker.create_sim()
        images = worker.run_sim()
        npy_to_gif(images, '/Desktop/video')


if __name__ == '__main__':
    run()
