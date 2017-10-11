## Information for Simulation Setup:



### 1st. Step: generate training data:

```cd /python_visual_mpc/visual_mpc_core/infrastructure/utility```
```python parallel_data_collection.py  <name_of_datafolder>```

Note: <name_of_datafolder> is the folder-name inside lsdc/pushing_data
Each of the folders in pushing_data must have a "/train" and "/test" subdirectory.
The hyperparams.py file to specifies how the data-collection is done.

### 2nd. Step: Train video prediction Model:
See Readme.md in python_visual_mpc/video_prediction

### 3rd Step: Run a benchmark on the pushing task:
```cd /python_visual_mpc/visual_mpc_core```

```python benchmarks.py <benchmark_folder_name>```

### Misc
In order to run rendering remotely on a different machine it is necessary to give remote users access to the x server:
Log into graphical session on remote computer and give access by typing:
xhost +

Also set the DISPLAY variable
export DISPLAY=:0

## Setup for using Rethink Sawyer:

### start kinect-bridge node:
cd visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/launch/bridgeonly.launch
./startkinect.sh

### start PD-Controller:
rosrun visual_mpc_rospkg joint_space_impedance.py


### start visual-mpc-client:
rosrun visual_mpc_rospkg visual_mpc_client.py <name_of_folder_inside:visual_mpc/experiments/cem_exp/benchmarks_sawyer>

### start visual-mpc-server (can be launched on newton4 or newton1):
rosrun visual_mpc_rospkg visual-mpc-server.py <name_of_folder_inside:visual_mpc/experiments/cem_exp/benchmarks_sawyer> --ngpu <number_of_gpus>
