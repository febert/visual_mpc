from python_visual_mpc.misc.html.util.html import HTML
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import read_scoes

import os

webpage = HTML('.','Intmstep_compare')


exp_base_dirs = ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/dur60',
                 '/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/dur60']

improvement_per_method = []
for basedir in exp_base_dirs:
    anglecost, improvement, score, sorted_ind = read_scoes(basedir)
    improvement_per_method.append(improvement)

tstep = 1
cem_iter = 2

nex = 100
ncols = len(exp_base_dirs) + 1 # first colmun is ntraj

filepaths = ['' for _ in range(ncols)]
strs = [''] + exp_base_dirs
webpage.add_images(filepaths, strs, filepaths)

for iex in range(nex):

    filepaths = ['']
    strs = ['traj{}'.format(iex)]
    for col in range(ncols-1):
        filepaths.append(exp_base_dirs[col] +'/verbose/traj{}/plan/directt{}iter_{}.gif'.format(iex, tstep, cem_iter))
        strs.append('improv: {}'.format(improvement_per_method[col][iex]))
    webpage.add_images(filepaths, strs, filepaths)

    filepaths = ['']
    strs = ['traj{}'.format(iex)]
    for col in range(ncols-1):
        filepaths.append(exp_base_dirs[col] +'/verbose/traj{}/video.gif'.format(iex))
        strs.append('')
    webpage.add_images(filepaths, strs, filepaths)

webpage.save(file='web/index1')
