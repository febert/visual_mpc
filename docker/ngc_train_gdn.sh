ngc batch run -n "Cloud-nv-us-west-2-111050"\
    -i "ucb_rail8888/tf1.4_gpu:based_nvidia" \
    --ace nv-us-west-2 \
    -in ngcv1 \
    -c "export VMPC_DATA_DIR=/workspace/pushing_data;
    export MUJOCO_PY_MJKEY_PATH=/workspace/visual_mpc/mujoco/mjpro131/mjkey.txt;
    export MUJOCO_PY_MJPRO_PATH=/workspace/visual_mpc/mujoco/mjpro131;
    export PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
    /bin/sleep 3600
    cd /workspace/visual_mpc/python_visual_mpc/goaldistancenet;
    python traingdn.py --hyper ../../tensorflow_data/gdn/tdac_cons0_cartgripper/conf.py --docker" \
    --result /results \
    --datasetid 8350:/workspace/pushing_data
