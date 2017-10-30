"""
Example script for using newton machines via docker + rllab
"""

import doodad as dd
import doodad.ssh as ssh
import doodad.mount as mount

MY_USERNAME = 'febert'

# Use local mode to test code
mode_local = dd.mode.LocalDocker(image='febert/tf1.3_gpu_com2')

# Use docker mode to launch jobs on newton machine
mode_ssh = dd.mode.SSHDocker(
    image='febert/tf1.3_gpu_com2',
    credentials=ssh.SSHCredentials(hostname='newton2.banatao.berkeley.edu',
                                   username='rail', identity_file='/home/frederik/.ssh/rail_lab_0617'),
    gpu=True
)

# Set up code and output directories


loc_conf_file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/cdna/conf.py'
mount_conf_file = '/docker_home/tensorflow_data/sawyer/cdna/conf.py'

loc_code_dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/python_visual_mpc'
mount_code_dir = '/docker_home/visual_mpc/python_visual_mpc'

rem_output_dir = '~/data/%s' % MY_USERNAME +'/visual_mpc/tensorflow_data/sawyer/cdna/modeldata'
mount_output_dir = '/docker_home/visual_mpc/tensorflow_data/sawyer/cdna/modeldata'  # this is the directory visible to the target script inside docker

rem_data_dir = '~/data/%s' % MY_USERNAME +'/visual_mpc/pushing_data'
mount_data_dir = '/docker_home/visual_mpc/pushing_data'

mounts = [
    mount.MountLocal(local_dir=loc_conf_file, mount_point=mount_conf_file),  # point to your rllab

    mount.MountLocal(local_dir=loc_code_dir, mount_point=mount_code_dir, pythonpath=True),  # point to your rllab
    # this output directory will be visible on the remote machine
    # TODO: this directory will have root permissions. For now you need to scp your data inside your script.
    # when output=True stuff will not be copied, the directory will only be added to the mounted directories
    mount.MountLocal(local_dir=rem_output_dir, mount_point=mount_output_dir, output=True),
    mount.MountLocal(local_dir=rem_data_dir, mount_point=mount_data_dir, output=True),
]

dd.launch_python(
    target='/home/frederik/Documents/catkin_ws/src/visual_mpc/python_visual_mpc/video_prediction/prediction_train_sawyer.py',  # point to a target script (absolute path).
    mode=mode_ssh,
    mount_points=mounts,
    # args='--hyper {}'.format(mount_conf_file),
    args={'hyper':mount_conf_file},
    target_mount_dir='/docker_home/visual_mpc/python_visual_mpc'
)
