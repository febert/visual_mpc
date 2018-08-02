# argument1: username, argument2: experiment
cd /home/$1/visual_mpc;
docker run  -v /home/$1/visual_mpc/pushing_data:/workspace/pushing_data \
                   -v /home/$1/visual_mpc:/mount \
-it \
sudeepdasari/mj_tf_cpu:firsttry "python python_visual_mpc/visual_mpc_core/parallel_data_collection.py $2"

