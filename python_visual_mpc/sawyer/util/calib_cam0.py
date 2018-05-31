
import numpy as np
from cvxpy import *

"""
Script used to find H and t for the transformation
x_Sawyer = H x_fcam + t
"""

# The ten calibration points in the fcam coordinate system
p_1_fcam = np.array([0.14712389, 0.09905503, 0.79387856]).reshape((3, 1))
p_2_fcam = np.array([-0.14546939, 0.18702181, 0.67870914]).reshape((3, 1))
p_3_fcam = np.array([-0.15560695, 0.08235386, 0.85817415]).reshape((3, 1))
p_4_fcam = np.array([0.09030248, 0.16380407, 0.85817415]).reshape((3, 1))
p_5_fcam = np.array([-0.19307526, 0.21289634, 0.67087269]).reshape((3, 1))
p_6_fcam = np.array([0.14809796, 0.0956221, 0.78997262]).reshape((3, 1))
p_7_fcam = np.array([-0.22354235, 0.00811434, 0.92185106]).reshape((3, 1))
p_8_fcam = np.array([-0.0294278, 0.20265216, 0.64906355]).reshape((3, 1))
p_9_fcam = np.array([0.05585613, 0.04350206, 0.87090842]).reshape((3, 1))
p_10_fcam = np.array([0.11689826, 0.00371152, 0.89444079]).reshape((3, 1))

p_fcam = [p_1_fcam, p_2_fcam, p_3_fcam, p_4_fcam, p_5_fcam, p_6_fcam, p_7_fcam, p_8_fcam, p_9_fcam, p_10_fcam]

# The ten corresponding calibration points in the Sawyer coordinate system
p_1_sawyer = np.array([0.62404614, 0.15892005, 0.18217241]).reshape((3, 1))
p_2_sawyer = np.array([0.77895433, -0.09286761, 0.18451509]).reshape((3, 1))
p_3_sawyer = np.array([0.6372649, -0.08373514, 0.18383977]).reshape((3, 1))
p_4_sawyer = np.array([0.77692706, 0.10950373, 0.18495063]).reshape((3, 1))
p_5_sawyer = np.array([0.82686237, -0.11465186, 0.18336545]).reshape((3, 1))
p_6_sawyer = np.array([0.649098, 0.17511494, 0.18841211]).reshape((3, 1))
p_7_sawyer = np.array([0.49541386, -0.17370474, 0.1830696]).reshape((3, 1))
p_8_sawyer = np.array([0.81347582, 0.00699111, 0.18454398]).reshape((3, 1))
p_9_sawyer = np.array([0.5533918, 0.09036484, 0.18636769]).reshape((3, 1))
p_10_sawyer = np.array([0.48335695, 0.13941328, 0.18298395]).reshape((3, 1))


p_sawyer = [p_1_sawyer, p_2_sawyer, p_3_sawyer, p_4_sawyer, p_5_sawyer, p_6_sawyer, p_7_sawyer, p_8_sawyer, p_9_sawyer, p_10_sawyer]

# Optimization variables
H = Variable(3, 3)
t = Variable(3)

# Optimization constraints
constraints = []

# Optimization objective
temp = []
for i in range(len(p_sawyer)):
        temp.append(norm(H * p_fcam[i] + t - p_sawyer[i]))
objective = Minimize(sum(temp))

# Solve optimization problem
prob = Problem(objective, constraints)
prob.solve()

print("H:\n", H.value)
print("t:\n", t.value)

np.save("H_fcam", H.value)
np.save("t_fcam", t.value)

H_fcam = np.load('H_fcam.npy')
t_fcam = np.load('t_fcam.npy')

error = 0
for p_f, p_s in zip(p_fcam, p_sawyer):
    pred_s = H_fcam.dot(p_f) + t_fcam.reshape((-1, 1))
    point_error = np.linalg.norm(pred_s - p_s)
    print('p_error', np.sqrt(point_error))
    print('pred_s: {} \t real_s: {}'.format(pred_s.reshape(-1), p_s.reshape(-1)))
    error += point_error
print('total_error', np.sqrt(error))