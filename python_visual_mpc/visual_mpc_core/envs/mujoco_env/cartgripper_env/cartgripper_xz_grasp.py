from .base_cartgripper import BaseCartgripperEnv


class CartgripperXZGrasp(BaseCartgripperEnv):
    def __init__(self, env_params, reset_state = None):
        super().__init__(env_params, reset_state)
        self._adim, self._sdim = 3, 3      # x z grasp


    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.set_hparam('filename', 'cartgripper_xz_grasp.xml')
        return parent_params