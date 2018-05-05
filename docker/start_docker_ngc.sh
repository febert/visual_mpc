# run image used for ngc
# argument1: username, argument2: $VMPC_DATA_DIR
nvidia-docker run   -v $VMPC_DATA_DIR:/mnt/pushing_data \
                    -v /home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data:/mnt/tensorflow_data \
                    -v /home/frederik/Documents/cloned_projects/video_prediction/pretrained_models:/mnt/pretrained_models \
                    -v /home/frederik/Documents/visual_mpc:/mount \
                    -v /home/frederik/Desktop:/Desktop \
-e VMPC_DATA_DIR=/mnt/pushing_data \
-e RESULT_DIR=/result \
-e TEN_DATA=/mnt/tensorflow_data \
-e ALEX_DATA=/mnt/pretrained_models \
-it \
nvcr.io/ucb_rail8888/tf_mj1.5 \
/bin/bash -c \
"/bin/bash"
