sudo nvidia-docker run  -v /mnt/sda1/pushing_data:/docker_home/pushing_data \
                        -v /home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/startgoal_shad/modeldata:/results \
-it -p 8888:8888 \
nvcr.io/ucb_rail8888/tf1.4_gpu:latest
/bin/bash -c \
"export VMPC_DATA_DIR=/docker_home/pushing_data;
export MUJOCO_PY_MJKEY_PATH=/docker_home/visual_mpc/mujoco/mjpro131/mjkey.txt;
export MUJOCO_PY_MJPRO_PATH=/docker_home/visual_mpc/mujoco/mjpro131;
/bin/bash"

