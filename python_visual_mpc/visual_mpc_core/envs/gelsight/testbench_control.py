import serial
import datetime
import cv2
from enum import Enum


# This class provides an interface to the gelsight testbench setup via serial.
class State(Enum):
    IDLE = 0
    BUSY = 1
    READY = 2


class TestBench:

    IDLE_MSGS = ["Initialized", "Moved", "Reset", "Pressed", "Ready"]

    def __init__(self, name, cam_index, side_cam_index=None):
        self.ser = serial.Serial(name, baudrate=250000, timeout=1)
        self.cap = cv2.VideoCapture(cam_index)
        if side_cam_index is not None:
            print('Setting up side cam')
            self.side_cam_cap = cv2.VideoCapture(side_cam_index)
        else:
            self.side_cam_cap = None
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.currmsg = ""
        self.state = State.IDLE

    def target_pos(self, x, y, z):
        """
        Command testbench to visit an xyz position.
        After calling target_pos, wait for the testbench to become idle again.
        """

        msg = 'x' + str(x) + 'y' + str(y) + 'z' + str(z) + '\n'
        self.ser.write(msg.encode())
        self.state = State.BUSY
        print(self.state)

    def reset(self):
        """
        Command testbench to reset using limit switches and reestablish
        the origin.
        After calling reset, wait for the testbench to become idle again.
        """

        self.ser.write(b'r\n')
        self.ser.flush()
        self.state = State.BUSY

    def flip_x_reset(self):
        self.ser.write(b'invx\n')
        self.ser.flush()

    def press_z(self, quick_steps, thresh):
        """
        Command testbench to descend in the z direction in small steps
        until the average threshold force is detected by the load cells.
        See TBControl::feedbackMoveZ, where the actual logic is. This is just
        a layer of serial communication.
        After calling press_z, wait for the testbench to become idle again.
        """

        msg = 'pz' + str(quick_steps) + 'w' + str(thresh) + '\n'
        self.ser.write(msg.encode())
        self.ser.flush()
        self.state = State.BUSY

    def reset_z(self):
        """
        Command testbench to reset the Z axis ONLY using the limit switch,
        re-establishing the origin.
        After calling reset_z, wait for the testbench to become idle again.
        """

        self.ser.write(b'rz\n')
        self.ser.flush()
        self.state = State.BUSY

    def busy(self):
        return self.state == State.BUSY

    def ready(self):
        return self.state == State.READY

    def start(self):
        """
        Command testbench to complete init sequence.
        This means resetting the axes and re-establishing the origin, as
        well as initializing and tareing the load cells.
        After calling start, wait for the testbench to become idle again.
        """

        self.ser.write(b'start\n')
        self.ser.flush()
        self.state = State.BUSY

    def __handle_msg(self, msg):
        pm = str(datetime.datetime.now()) + ": " + msg
        if any([msg.startswith(key) for key in self.IDLE_MSGS]):
            self.state = State.IDLE
        if msg.startswith("Starting"):
            self.state = State.READY
        print(pm)
        return pm

    def update(self):
        """
        If you are waiting on a message (for example, indicator that state will
        change from busy to idle), call update in a loop, otherwise new messages
        will not be received over serial.
        """

        for i in range(self.ser.inWaiting()):
            ch = self.ser.read().decode()
            if ch == "\n":
                self.__handle_msg(self.currmsg)
                self.currmsg = ""
            else:
                self.currmsg += ch
        # Keep camera buffer empty
        self.cap.grab()
        if self.side_cam_cap:
            self.side_cam_cap.grab()

    def get_side_cam_frame(self):
        if self.side_cam_cap is None:
            return None
        ret, frame = self.side_cam_cap.read()
        assert ret, "Failed to get side cam frame"
        return frame

    def get_frame(self):
        ret, frame = self.cap.read()
        assert ret, "Failed to get frame"
        return frame

    def req_data(self):
        """
        Queries testbench for latest XYZ position and load cell readings.
        This method, unlike other commands sent to the testbench,
        is _synchronous_. The state does not change to busy, and the return
        value is a dictionary with parsed values.
        """

        self.ser.write(b'l\n')
        self.ser.flush()
        data = self.ser.readline()
        while data.decode().startswith('l'):  # Ignore echo of log request
            data = self.ser.readline()
        return self.__parse_data_str(data.decode())

    def __parse_data_str(self, data):
        """
        Turn data strings from testbench into usable dictionaries
        """
        res = dict()
        res['x'] = int(data[data.find('X') + 3:data.find('Y')])
        res['y'] = int(data[data.find('Y') + 3:data.find('Z')])
        data = data[data.find('Z') + 3:]
        res['z'] = int(data[:data.find(' ')])
        data = data[data.find(':') + 2:]
        for i in range(4):
            res['force_' + str(i + 1)] = float(data[:data.find(' ')])
            if i < 3:
                data = data[data.find(' ') + 4:]
        return res

    def frame_shape(self):
        return self.height, self.width
