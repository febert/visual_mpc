# argument1: username, argument2: $VMPC_DATA_DIR
nvidia-docker run  -v $2:/workspace/pushing_data \
                   -v /home/sudeep/Documents/visual_mpc:/mount \
                   -v /home/sudeep/Desktop:/Desktop \
-it \
sudeep/mj_tf bash

