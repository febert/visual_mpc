import rospy
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.visual_mpc_rospkg.src.utils.robot_controller import RobotController
from wsg_50_common.msg import Cmd, Status
from threading import Semaphore, Lock
import numpy as np
from python_visual_mpc.visual_mpc_core.envs.util.interpolation import CSpline
import python_visual_mpc.visual_mpc_core.envs.sawyer_robot.visual_mpc_rospkg as visual_mpc_rospkg
from intera_core_msgs.msg import JointCommand
import cPickle as pickle
import intera_interface


# constants for robot control
NEUTRAL_JOINT_ANGLES = np.array([0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, -1.11748145])
MAX_TIMEOUT = 30
DURATION_PER_POINT = 0.01
N_JOINTS = 7
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])
GRIPPER_CLOSE = 6   # chosen so that gripper closes entirely without pushing against itself
GRIPPER_OPEN = 96   # chosen so that gripper opens entirely without pushing against outer rail

class ImpedanceWSGController(RobotController):
    def __init__(self, control_rate, robot_name):
        self.max_release = 0
        RobotController.__init__(self)
        self.sem_list = [Semaphore(value = 0)]
        self._status_mutex = Lock()
        self.robot_name = robot_name

        self._desired_gpos = GRIPPER_OPEN
        self.gripper_speed = 300

        self._force_counter = 0
        self._integrate_gripper_force = 0.
        self.num_timeouts = 0

        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        self.gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        rospy.Subscriber("/wsg_50_driver/status", Status, self._gripper_callback)

        print("waiting for first status")
        self.sem_list[0].acquire()
        print('gripper initialized!')

        self.control_rate = rospy.Rate(control_rate)

        self._navigator = intera_interface.Navigator()
        self._navigator.register_callback(self._close_gripper_handler, 'right_button_ok')

    def _close_gripper_handler(self, value):
        if value:
            self.close_gripper()    #close gripper on button release

    def set_gripper_speed(self, new_speed):
        assert new_speed > 0 and new_speed <= 600, "Speed must be in range (0, 600]"
        self.gripper_speed = new_speed

    def get_gripper_status(self, integrate_force=False):
        self._status_mutex.acquire()
        cum_force, cntr = self._integrate_gripper_force, self._force_counter
        width, force = self.gripper_width, self.gripper_force
        self._integrate_gripper_force = 0.
        self._force_counter = 0
        self._status_mutex.release()

        if integrate_force and cntr > 0:
            print("integrating with {} readings, cumulative force: {}".format(cntr, cum_force))
            return width, cum_force / cntr

        return width, force

    def get_limits(self):
        return GRIPPER_CLOSE, GRIPPER_OPEN

    def open_gripper(self, wait = False):
        self.set_gripper(GRIPPER_OPEN, wait = wait)

    def close_gripper(self, wait = False):
        self.set_gripper(GRIPPER_CLOSE, wait = wait)

    def _set_gripper(self, command_pos, wait = False):
        self._desired_gpos = command_pos
        if wait:
            if self.num_timeouts > MAX_TIMEOUT:
                rospy.signal_shutdown("MORE THAN {} GRIPPER TIMEOUTS".format(MAX_TIMEOUT))

            sem = Semaphore(value=0)  # use of semaphore ensures script will block if gripper dies during execution

            self._status_mutex.acquire()
            self.sem_list.append(sem)
            self._status_mutex.release()

            start = rospy.get_time()
            print("gripper sem acquire, list len-{}".format(len(self.sem_list)))
            sem.acquire()
            print("waited on gripper for {} seconds".format(rospy.get_time() - start))

    def set_gripper(self, command_pos, wait = False):
        assert command_pos >= GRIPPER_CLOSE and command_pos <= GRIPPER_OPEN, "Command pos must be in range [GRIPPER_CLOSE, GRIPPER_OPEN]"
        self._set_gripper(command_pos, wait = wait)

    def _gripper_callback(self, status):
        # print('callback! list-len {}, max_release {}'.format(len(self.sem_list), self.max_release))
        self._status_mutex.acquire()

        self.gripper_width, self.gripper_force = status.width, status.force
        self._integrate_gripper_force += status.force
        self._force_counter += 1

        cmd = Cmd()
        cmd.pos = self._desired_gpos
        cmd.speed = self.gripper_speed

        self.gripper_pub.publish(cmd)

        if len(self.sem_list) > 0:
            gripper_close = np.isclose(self.gripper_width, self._desired_gpos, atol=1e-1)

            if gripper_close or self.gripper_force > 0 or self.max_release > 15:
                if self.max_release > 15:
                    self.num_timeouts += 1
                for s in self.sem_list:
                    s.release()
                self.sem_list = []

            self.max_release += 1      #timeout for when gripper responsive but can't acheive commanded state
        else:
            self.max_release = 0

        self._status_mutex.release()

    def neutral_with_impedance(self, duration=2):
        waypoints = [NEUTRAL_JOINT_ANGLES]
        self.move_with_impedance(waypoints, duration)

    def move_with_impedance(self, waypoints, duration=1.5):
        """
        Moves from curent position to final position while hitting waypoints
        :param waypoints: List of arrays containing waypoint joint angles
        :param duration: trajectory duration
        """

        jointnames = self.limb.joint_names()
        prev_joint = np.array([self.limb.joint_angle(j) for j in jointnames])
        waypoints = np.array([prev_joint] + waypoints)

        spline = CSpline(waypoints, duration)

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        time = rospy.get_time()
        while time < finish_time:
            pos, velocity, acceleration = spline.get(time - start_time)
            command = JointCommand()
            command.mode = JointCommand.POSITION_MODE
            command.names = jointnames
            command.position = pos
            command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
            command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
            self._cmd_publisher.publish(command)

            self.control_rate.sleep()
            time = rospy.get_time()

        for i in xrange(10):
            command = JointCommand()
            command.mode = JointCommand.POSITION_MODE
            command.names = jointnames
            command.position = waypoints[-1]
            self._cmd_publisher.publish(command)

            self.control_rate.sleep()

    def redistribute_objects(self):
        self.set_neutral()
        print('redistribute...')

        file = '/'.join(str.split(visual_mpc_rospkg.__file__, "/")[
                        :-1]) + '/src/utils/pushback_traj_{}.pkl'.format(self.robot_name)

        self.joint_pos = pickle.load(open(file, "rb"))

        replay_rate = rospy.Rate(700)
        for t in range(len(self.joint_pos)):
            print('step {0} joints: {1}'.format(t, self.joint_pos[t]))
            replay_rate.sleep()
            self.move_with_impedance([self.joint_pos[t]])
