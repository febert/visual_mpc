ngc batch run -n "Cloud-nv-us-west-2-111050"\
-i "ucb_rail8888/tf1.4_gpu:based_nvidia" \
--ace nv-us-west-2 \
-in ngcv1 \
-c "export VMPC_DATA_DIR=/workspace/pushing_data ;
    export PATH=/opt/conda/bin:$PATH;
cd /workspace/visual_mpc/python_visual_mpc/goaldistancenet;
python traingdn.py --hyper ../../tensorflow_data/gdn/tdac_cons0_instn/conf.py --docker" \
--result /results \
--datasetid 8350:/workspace/pushing_data
