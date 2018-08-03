import numpy as np

def is_touching(finger_sensors, touchthresh = 0):
    return np.max(finger_sensors) > touchthresh