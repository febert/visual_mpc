#!/usr/bin/python
import rospy
import numpy as np
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_controller import RobotController

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
from sensor_msgs.msg import JointState
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.primitives_regintervals import quat_to_zangle, zangle_to_quat
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils import inverse_kinematics
import pdb
def get_endeffector_pose(limb, fksvc, name_of_service):
    fkreq = SolvePositionFKRequest()
    joints = JointState()
    joints.name = limb.joint_names()
    joints.position = [limb.joint_angle(j)
                       for j in joints.name]

    # Add desired pose for forward kinematics
    fkreq.configuration.append(joints)
    fkreq.tip_names.append('right_hand')
    try:
        rospy.wait_for_service(name_of_service, 5)
        resp = fksvc(fkreq)
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

def state_to_angles(target_state, limb):
    quat = zangle_to_quat(target_state[3])
    desired_pose = inverse_kinematics.get_pose_stamped(target_state[0],
                                                       target_state[1],
                                                       target_state[2],
                                                       quat)

    start_joints = limb.joint_angles()
    try:
        target_ja = inverse_kinematics.get_joint_angles(desired_pose, seed_cmd=start_joints,
                                                        use_advanced_options=True)
    except ValueError:
        rospy.logerr('no inverse kinematics solution found')
        return None
    return target_ja
def main():
    controller = RobotController()
    controller.set_neutral()

    name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
    fksvc = rospy.ServiceProxy(name_of_service, SolvePositionFK)

    pos = get_endeffector_pose(controller.limb, fksvc, name_of_service)
    print 'robot xyz', pos[:3]
    print 'zangle', quat_to_zangle(pos[3:]) / np.pi

    angles = np.linspace(0.5 * np.pi, np.pi, 100)
    for theta in angles:
        print 'attempting {}'.format(theta / np.pi)
        target = np.array([pos[0], pos[1], pos[2], theta])
        ja = state_to_angles(target, controller.limb)
        if ja is not None:
            controller.set_joints(ja)
        else:
            print 'cannnot hit {}'.format(theta)

if __name__ == '__main__':
    main()