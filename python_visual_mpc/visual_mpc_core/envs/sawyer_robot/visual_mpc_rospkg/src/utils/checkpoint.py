import os

def write_ckpt(checkpoint_file, tr, i_grp):
    with open(checkpoint_file, 'w') as f:
        f.write("last trajectory, current group%f \n")
        f.write("{} {} \n".format(tr, i_grp))


def write_timing_file(save_dir, avg, avg_nstep, nfail_traj):
    with open(os.path.join(save_dir, 'timingfile.txt'), 'w') as f:
        f.write(
            "average duration for trajectory (including amortization of redistribution trajectory): {} \n".format(avg))
        f.write("expected number of trajectory per day: {} \n".format(24. * 3600. / avg))
        f.write("number of failed trajectories within {} trials: {}\n".format(avg_nstep, nfail_traj))


def parse_ckpt(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        i = 0
        for line in f:
            if i == 1:
                numbers_str = line.split()
                itr, igrp = [int(x) for x in numbers_str]  # map(float,numbers_str) works t
            i += 1
        return itr, igrp