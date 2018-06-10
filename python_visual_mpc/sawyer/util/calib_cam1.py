
import numpy as np
from cvxpy import *

"""
Script used to find H and t for the transformation
x_Sawyer = H x_lcam + t
"""

# The ten calibration points in the lcam coordinate system
p_1_lcam = np.array([0.04664278, -0.02472251, 0.80153617]).reshape((3, 1))
p_2_lcam = np.array([0.21279803, 0.13661773, 0.62195701]).reshape((3, 1))
p_3_lcam = np.array([0.01090158, 0.1299757, 0.54170024]).reshape((3, 1))
p_4_lcam = np.array([0.163212449, 0.000538726007, 0.768352291]).reshape((3, 1))
p_5_lcam = np.array([0.25212912, 0.16199255, 0.59423369]).reshape((3, 1))
p_6_lcam = np.array([0.04300808, -0.02641973, 0.82065621]).reshape((3, 1))
p_7_lcam = np.array([-0.11376011, 0.19173899, 0.55449037]).reshape((3, 1))
p_8_lcam = np.array([0.24237161, 0.06557262, 0.69774114]).reshape((3, 1))
p_9_lcam = np.array([-0.05169478, 0.03228137, 0.73730863]).reshape((3, 1))
p_10_lcam = np.array([-0.12158303, -0.00191091, 0.71539879]).reshape((3, 1))
p_11_lcam = np.array([0.20548082, 0.06576004, 0.69871352]).reshape((3, 1))
p_12_lcam = np.array([-0.04660003, -0.00742236, 0.4811249]).reshape((3, 1))
p_13_lcam = np.array([0.12736719, -0.11713373, 0.60629696]).reshape((3, 1))
p_14_lcam = np.array([0.21898202, 0.10598259, 0.49980709]).reshape((3, 1))
p_15_lcam = np.array([-0.06648953, 0.10429062, 0.47103609]).reshape((3, 1))
p_16_lcam = np.array([0.16626607, -0.12558073, 0.61680587]).reshape((3, 1))
p_17_lcam = np.array([-0.1053655, -0.07381605, 0.75874501]).reshape((3, 1))

p_lcam = [p_1_lcam, p_2_lcam, p_3_lcam, p_4_lcam, p_5_lcam, p_6_lcam, p_7_lcam, 
          p_8_lcam, p_9_lcam, p_10_lcam, p_11_lcam, p_12_lcam, p_13_lcam, p_14_lcam,
          p_15_lcam, p_16_lcam, p_17_lcam]

# The 17 corresponding calibration points in the Sawyer coordinate system
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
p_11_sawyer = np.array([0.77663933, 0.00884187, 0.17993153]).reshape((3, 1))
p_12_sawyer = np.array([0.56730004, -0.11262669, 0.35859245]).reshape((3, 1))
p_13_sawyer = np.array([0.72347009, 0.04582043, 0.35488332]).reshape((3, 1))
p_14_sawyer = np.array([0.79464528, -0.1679792, 0.25699683]).reshape((3, 1))
p_15_sawyer = np.array([0.55278069, -0.16831283, 0.26525751]).reshape((3, 1))
p_16_sawyer = np.array([0.73617149, 0.06522711, 0.35470406]).reshape((3, 1))
p_17_sawyer = np.array([0.49645224, 0.17386133, 0.23895355]).reshape((3, 1))


p_sawyer = [p_1_sawyer, p_2_sawyer, p_3_sawyer, p_4_sawyer, p_5_sawyer, p_6_sawyer, p_7_sawyer, 
            p_8_sawyer, p_9_sawyer, p_10_sawyer, p_11_sawyer, p_12_sawyer, p_13_sawyer, p_14_sawyer,
            p_15_sawyer,p_16_sawyer,p_17_sawyer]

# Optimization variables
H = Variable(3, 3)
t = Variable(3)

# Optimization constraints
constraints = []

# Optimization objective
temp = []
for i in range(len(p_sawyer)):
        temp.append(norm(H * p_lcam[i] + t - p_sawyer[i]))
objective = Minimize(sum(temp))

# Solve optimization problem
prob = Problem(objective, constraints)
prob.solve()

print("H:\n", H.value)
print("t:\n", t.value)

np.save("H_lcam", H.value)
np.save("t_lcam", t.value)

H_lcam = np.load('H_lcam.npy')
t_lcam = np.load('t_lcam.npy')

error = 0
for p_l, p_s in zip(p_lcam, p_sawyer):
    pred_s = H_lcam.dot(p_l) + t_lcam.reshape((-1, 1))
    point_error = np.linalg.norm(pred_s - p_s)
    print('p_error', np.sqrt(point_error))
    print('pred_s: {} \t real_s: {}'.format(pred_s.reshape(-1), p_s.reshape(-1)))
    error += point_error
print('total_error', np.sqrt(error))