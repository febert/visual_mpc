#based on Dockerfile from mujoco-py
FROM mjpy

COPY ./requirements.txt /mujoco_py/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
