# argument1: username, argument2: $VMPC_DATA_DIR
nvidia-docker run  -v $VMPC_DATA_DIR:/workspace/pushing_data \
                   -v /home/$USER/Documents/visual_mpc:/mount/visual_mpc \
                   -v /home/$USER/Documents/cloned_projects/video_prediction:/mount/video_prediction \
-it \
febert/tf_mj1.5:runmount \
/bin/bash -c \
"/bin/bash"

