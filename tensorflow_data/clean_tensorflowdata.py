# go through modeldata directories and delete model* files which are have smaller iterations
#than the maximum one

import os
import re
import numpy as np
import pdb
current_dir = os.path.dirname(os.path.realpath(__file__))


dryrun = False
for path, subdirs, files in os.walk(current_dir):
    # print path
    # print subdirs
    # print files

    iter_numbers = []
    for file in files:

        if "model" in file:
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
        max_iter = np.max(np.array(iter_numbers))
        iter_numbers.remove(max_iter)

        if iter_numbers != []:
            print('max iternumber found in {}:{}'.format(path, max_iter))
            print('going to delete: {}'.format(iter_numbers))

            for n in iter_numbers:
                try:
                    if not dryrun:
                        os.remove(os.path.join(path, 'model' + str(n)) + '.data-00000-of-00001')
                        os.remove(os.path.join(path,'model' +str(n)) +'.index')
                        os.remove(os.path.join(path,'model' +str(n)+'.meta'))
                except:
                    print('not able to remove')