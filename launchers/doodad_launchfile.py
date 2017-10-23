"""
Example script for using newton machines via docker + rllab
"""

import doodad as dd
import doodad.ssh as ssh
import doodad.mount as mount

MY_USERNAME = 'febert'

# Use local mode to test code
mode_local = dd.mode.LocalDocker(
    image='tf1.3_gpu'
)

# Use docker mode to launch jobs on newton machine
mode_ssh = dd.mode.SSHDocker(
    image='tf1.3_gpu',
    credentials=ssh.SSHCredentials(hostname='newton2.banatao.berkeley.edu',
                                   username='rail', identity_file='path/to/identity'),
)

# Set up code and output directories


loc_conf_dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/cdna/conf.py'
rem_conf_dir = '/docker_home/tensorflow_data/sawyer/cdna'

rem_output_dir = rem_conf_dir + '/modeldata'  # this is the directory visible to the target script inside docker

mounts = [
    mount.MountLocal(local_dir=loc_conf_dir, mount_point=rem_conf_dir + '/conf.py'),  # point to your rllab

    # this output directory will be visible on the remote machine 
    # TODO: this directory will have root permissions. For now you need to scp your data inside your script.
    mount.MountLocal(local_dir='~/data/%s' % MY_USERNAME +rem_output_dir, mount_point=rem_output_dir, output=True),
]

dd.launch_python(
    target='/home/frederik/Documents/catkin_ws/src/visual_mpc/python_visual_mpc/video_prediction/prediction_train_sawyer.py',  # point to a target script (absolute path).
    mode=mode_ssh,
    mount_points=mounts,
    args='--hyper ../../tensorflow_data/sawyer/cdna/conf.py'
)
