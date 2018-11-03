from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred


class GoalImageController(CEM_Controller_Vidpred):
    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None, goal_image=None):
        self.goal_image = goal_image
        return super(GoalImageController, self).act(t, i_tr, desig_pix, goal_pix, images, state)
