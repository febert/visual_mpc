

objects = [5, 18 , 19 , 88, 90]

startgoalconfs = [1,2,3,4]

path = 'weissgripper_regstartgoal_tradeoff'

file = 'benchmark_schedule.txt'

with open(file, 'w') as f:
    for ob in objects:
        for sg in startgoalconfs:
            f.write('ob{}s{}\n'.format(ob, sg))