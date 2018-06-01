from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_controller import RobotController
import rospy

from wsg_50_common.srv import Move
from wsg_50_common.msg import Cmd, Status
from threading import Semaphore

GRIPPER_CLOSE = 4   #chosen so that gripper closes entirely without pushing against itself
GRIPPER_OPEN = 95   #chosen so that gripper opens entirely without pushing against outer rail

class WSGRobotController(RobotController):
    def __init__(self):
        RobotController.__init__(self)
        self.first_status = False
        self.status_sem = Semaphore(value = 0)

        rospy.Subscriber("/wsg_50_driver/status", Status, self.gripper_callback)
        self.gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        self._desired_gpos = GRIPPER_OPEN
        self.gripper_speed = 300

        print("waiting for first status")
        self.status_sem.acquire()
        print('gripper initialized!')

    def set_gripper_speed(self, new_speed):
        assert new_speed > 0 and new_speed <= 600, "Speed must be in range (0, 600]"
        self.gripper_speed = new_speed

    def get_gripper_status(self):
        return self.gripper_width, self.gripper_force

    def get_limits(self):
        return GRIPPER_CLOSE, GRIPPER_OPEN

    def open_gripper(self):
        self.set_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.set_gripper(GRIPPER_CLOSE)

    def set_gripper(self, command_pos):
        assert command_pos >= GRIPPER_CLOSE and command_pos <= GRIPPER_OPEN, "Command pos must be in range [GRIPPER_CLOSE, GRIPPER_OPEN]"
        self._desired_gpos = command_pos

    def gripper_callback(self, status):
        self.gripper_width, self.gripper_force = status.width, status.force

        cmd = Cmd()
        cmd.pos = self._desired_gpos
        cmd.speed = self.gripper_speed

        self.gripper_pub.publish(cmd)

        if not self.first_status:
            self.first_status = True
            self.status_sem.release()