import ipdb
import matplotlib.pyplot as plt
import numpy as np


runs=[]
runs.append({'name': 'fix0.25'   , 'jobid': 54168 , 'l1err' : 0.03088 , 'meanstep': 3.21})
runs.append({'name': 'fix0.50'   , 'jobid': 54167 , 'l1err' : 0.03249 , 'meanstep': 6.46})
runs.append({'name': 'fix0.75'   , 'jobid': 54169 , 'l1err' : 0.04392 , 'meanstep': 6.47})
runs.append({'name': 'fix1.00'   , 'jobid': 54186 , 'l1err' : 0.04409 , 'meanstep': 7.92})
runs.append({'name': 'genmin0.5' , 'jobid': 50779 , 'l1err' : 0.03388 , 'meanstep': 8.43})
runs.append({'name': 'genmin2'   , 'jobid': 53977 , 'l1err' : 0.03139 , 'meanstep': 7.27})
runs.append({'name': 'genmin4'   , 'jobid': 53978 , 'l1err' : 0.02833 , 'meanstep': 5.64})
runs.append({'name': 'genmin7'   , 'jobid': 53979 , 'l1err' : 0.02757 , 'meanstep': 4.89})
runs.append({'name': 'genmin10'  , 'jobid': 53794 , 'l1err' : 0.02574 , 'meanstep': 3.82})
runs.append({'name': 'genmin30'  , 'jobid': 53795 , 'l1err' : 0.02146 , 'meanstep': 2.55})
runs.append({'name': 'min', 'jobid': 50774 , 'l1err' : 0.02119 , 'meanstep': 2.16})

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
fig, ax = plt.subplots(1, 1, figsize=(3,2))

x = np.array([run['meanstep'] for run in runs])
y = np.array([run['l1err'] for run in runs])
names = np.array([run['name'] for run in runs])
ours = np.array(['min' in run['name'] or 'gen' in run['name'] for run in runs])


def labeled_scatter(ax, x, y, label, offset=-0.003, **args):
    ax.scatter(x, y, **args)
    for i, txt in enumerate(label):
        ax.annotate(txt, (x[i]-0.5, y[i]+offset), fontsize=12)

labeled_scatter(ax,
    x[np.logical_not(ours)],
    y[np.logical_not(ours)],
    names[np.logical_not(ours)],
    offset=+0.003,
    c='blue',
    marker='o'
)

ax.plot(x[ours], y[ours], c='red', marker='s', label='ours')
# ax.annotate('genmin.5', (x[ours][0]-0.75, y[ours][0]-0.005), fontsize=12)
# ax.annotate('min', (x[ours][-1], y[ours][-1]-0.005), fontsize=12)
# labeled_scatter(ax,
#     x[ours],
#     y[ours],
#     names[ours],
#     c='red',
#     marker='s'
# )

ax.set_xlabel('match-step')
ax.set_ylabel('min $\ell_1$ err')
ax.legend(loc='upper left', fontsize=14)
ax.set_ylim([0.015, 0.05])

plt.savefig('plots/grasp_fwdpred_scatter.png', bbox_inches='tight', dpi = 300)
plt.show()
