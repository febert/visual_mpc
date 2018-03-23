nvidia-docker run  -v /mnt/sda1/pushing_data:/workspace/pushing_data \
                   -v /home/frederik/Documents/catkin_ws/src/visual_mpc:/mount \
                   -v /home/frederik/Desktop:/Desktop \
-it \
tf_mj1.5 \
/bin/bash -c \
"export VMPC_DATA_DIR=/workspace/pushing_data;
/bin/bash"

