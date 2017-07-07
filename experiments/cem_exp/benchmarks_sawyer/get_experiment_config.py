import numpy as np




motion_types = ['sd', 'ld', 'clt', 'multp', 'wp']


objects = ['2,5','6,7']

setup1 = [1, 2]

run_numbers = range(2)

experiment_name = 'multobj_confs'
path = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg/'


with open(path +  experiment_name+ '/' + experiment_name +'.txt', 'w') as f:

    for obj in objects:
        for s in setup1:
            for r in run_numbers:
                name = "run{}setup{}_objects{}".format(r,s, obj)
                print name
                f.write(experiment_name + '/' +name + '\n')