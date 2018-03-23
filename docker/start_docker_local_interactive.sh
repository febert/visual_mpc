nvidia-docker run  -v /mnt/sda1/pushing_data:/workspace/pushing_data \
                        -v /home/frederik/Desktop:/outputs \
-it -p 8888:8888 \
nvcr.io/ucb_rail8888/tf1.4_gpu:based_nvidia \
/bin/bash -c \
"export VMPC_DATA_DIR=/workspace/pushing_data;
export LANG C.UTF-8
export PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
/bin/bash"

