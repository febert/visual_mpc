from python_visual_mpc.visual_mpc_core.envs.sawyer_sim.base_sawyer import BaseSawyerEnv

class VanillaSawyerEnv(BaseSawyerEnv):
    def __init__(self, hyper):
        super().__init__(**hyper)