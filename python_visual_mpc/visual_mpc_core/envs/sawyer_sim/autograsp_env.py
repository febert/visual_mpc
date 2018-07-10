from python_visual_mpc.visual_mpc_core.envs.sawyer_sim.base_sawyer import BaseSawyerEnv

class AutograspSawyerEnv(BaseSawyerEnv):
    def __init__(self, hyper):
        super().__init__(**hyper)