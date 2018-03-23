#based on Dockerfile from mujoco-py
FROM mjpy

COPY ./requirements.txt /mujoco_py/
RUN pip install -r requirements.txt

#ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64
#RUN wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.2/prod/9.0_20180316/cudnn-9.0-linux-x64-v7.1


