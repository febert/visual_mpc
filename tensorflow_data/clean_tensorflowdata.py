# go through modeldata directories and delete model* files which are have smaller iterations
#than the maximum one

import os
import re
import numpy as np
import pdb
current_dir = os.path.dirname(os.path.realpath(__file__))

for path, subdirs, files in os.walk(current_dir):
    # print path
    # print subdirs
    # print files

    iter_numbers = []
    for file in files:

        if "model" in file:
            iter_num = re.match('.*?([0-9]+)$', file)
            if iter_num == None:
                continue
            else:
                iter_numbers.append(int(iter_num.group(1)))
    if iter_numbers != []:
        max_iter = np.max(np.array(iter_numbers))
        iter_numbers.remove(max_iter)

        if iter_numbers != []:
            print 'max iternumber found in {}:{}'.format(path, max_iter)
            print 'going to delete: {}'.format(iter_numbers)

            for n in iter_numbers:
                try:
                    os.remove(os.path.join(path,'model' +str(n)))
                    os.remove(os.path.join(path,'model' +str(n)+'.meta'))
                except:
                    print 'not able to remove'