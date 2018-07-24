import rospy
import numpy as np
from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
from sensor_msgs.msg import JointState


class LimbRecorder:
    def __init__(self, control_limb):
        self._limb = control_limb
        self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

    def get_state(self):
        joint_angles = self.get_joint_angles()
        joint_velocities = self.get_joint_angles_velocity()
        eep = self.get_endeffector_pose()

        return joint_angles, joint_velocities, eep

    def get_joint_names(self):
        return self._limb.joint_names()

    def get_joint_cmd(self):
        return self._limb.joint_angles()

    def get_joint_angles(self):
        return np.array([self._limb.joint_angle(j) for j in self._limb.joint_names()])

    def get_joint_angles_velocity(self):
        return np.array([self._limb.joint_velocity(j) for j in self._limb.joint_names()])

    def get_endeffector_pose(self):
        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self._limb.joint_names()
        joints.position = [self._limb.joint_angle(j)
                           for j in joints.name]

        # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
        fkreq.tip_names.append('right_hand')
        try:
            rospy.wait_for_service(self.name_of_service, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                        resp.pose_stamp[0].pose.position.y,
                        resp.pose_stamp[0].pose.position.z,
                        resp.pose_stamp[0].pose.orientation.x,
                        resp.pose_stamp[0].pose.orientation.y,
                        resp.pose_stamp[0].pose.orientation.z,
                        resp.pose_stamp[0].pose.orientation.w])

        return pos


class LimbWSGRecorder(LimbRecorder):
    def __init__(self, wsg_controller):
        self._ctrl = wsg_controller
        LimbRecorder.__init__(self, wsg_controller.limb)

    def get_gripper_state(self):
        g_width, g_force = self._ctrl.get_gripper_status(integrate_force=True)
        close_thresh, open_thresh = self._ctrl.get_limits()

        gripper_status = (g_width - close_thresh) / (open_thresh - close_thresh)  #t = 1 --> open, and t = 0 --> closed

        return np.array([gripper_status]), np.array([g_force])

    def get_state(self):
        gripper_state, force_sensor = self.get_gripper_state()
        j_angles, j_vel, eep = LimbRecorder.get_state(self)
        return j_angles, j_vel, eep, gripper_state, force_sensor
