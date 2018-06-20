# go through modeldata directories and delete model* files which are have smaller iterations
#than the maximum one

import os
import sys
import re
import numpy as np
import pdb
# current_dir = os.path.dirname(os.path.realpath(__file__))
current_dir = '/mnt/sda1/experiments/cem_exp/benchmarks/alexmodel'


dryrun = False
for path, subdirs, files in os.walk(current_dir):
    # print path
    # print subdirs
    # print files

    iter_numbers = []
    for file in files:

        if "directt" in file:
            # iter_num = re.match('.*?([0-9]+)$', file)
            try:
                iter_num = re.search(r'\d+', file).group()
                if iter_num == None:
                    continue
                else:
                    if int(iter_num) not in iter_numbers:
                        iter_numbers.append(int(iter_num))
            except:
                print("could not read internum in", file)

    if iter_numbers != []:
        min_num = np.min(np.array(iter_numbers))
        iter_numbers.remove(min_num)

        if iter_numbers != []:
            print('going to delete: {}'.format(iter_numbers))

            for n in iter_numbers:
                # try:
                cmd = 'rm ' + os.path.join(path, 'directt{}*'.format(n))
                print(cmd)
                if not dryrun:
                    os.system(cmd)
                # except:
                #     print('not able to remove')