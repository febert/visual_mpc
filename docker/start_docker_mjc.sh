nvidia-docker run  -v /mnt/sda1/pushing_data:/workspace/pushing_data \
                   -v /home/$USER/Documents/catkin_ws/src/visual_mpc:/mount \
                   -v /home/$USER/Desktop:/Desktop \
-it \
nvcr.io/ucb_rail8888/tf_mj1.5 \
/bin/bash -c \
"export VMPC_DATA_DIR=/workspace/pushing_data;
/bin/bash"

