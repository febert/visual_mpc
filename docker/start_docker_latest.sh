# argument1: username
# argument2: $VMPC_DATA_DIR
# argument3: $RESULT_DIR
nvidia-docker run -v $2:/workspace/pushing_data \
			-v /home/$1/Documents/catkin_ws/src/visual_mpc:/workspace/visual_mpc \
			-v $3:/result \
			-v /home/$1/Desktop:/Desktop \
-e VMPC_DATA_DIR=/workspace/pushing_data \
-e RESULT_DIR=/result \
-e TEN_DATA=/workspace/visual_mpc/tensorflow_data \
-it febert/tf_mj1.5:latest \
/bin/bash -c \
"/bin/bash"
