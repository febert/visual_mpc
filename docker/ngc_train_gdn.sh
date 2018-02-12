ngc batch run -n "Cloud-nv-us-west-2-111050"\
-i "ucb_rail8888/tf1.4_gpu:latest" \
--ace nv-us-west-2 \
-in ngcv1 \
-c "export VMPC_DATA_DIR=/docker_home/pushing_data ;
cd /docker_home/visual_mpc/python_visual_mpc/goaldistancenet;
python traingdn.py --hyper ../../tensorflow_data/gdn/startgoal_shad/conf.py --docker" \
--result /results \
--datasetid 8350:/docker_home/pushing_data
