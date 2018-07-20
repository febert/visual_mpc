import numpy as np

def is_touching(finger_sensors):
    return np.max(finger_sensors) > 0