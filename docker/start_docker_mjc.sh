# argument1: username, argument2: $VMPC_DATA_DIR
nvidia-docker run  -v $2:/workspace/pushing_data \
                   -v /home/$1/Documents/catkin_ws/src/visual_mpc:/mount/visual_mpc \
                   -v /home/$1/Documents/cloned_projects/video_prediction:/mount/video_prediction \
-it \
febert/tf_mj1.5:runmount \

/bin/bash -c \
"/bin/bash"

