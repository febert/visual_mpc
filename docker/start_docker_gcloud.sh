# argument1: username, argument2: $VMPC_DATA_DIR
docker run  -v /home/sudeep/visual_mpc/pushing_data:/workspace/pushing_data \
                   -v /home/sudeep/visual_mpc:/mount \
-it \
sudeepdasari/mj_tf_cpu:firsttry bash

