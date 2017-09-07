# How to start visual-MPC (TODO: put all of this in one or sevearl launchfiles!)

#start kinect-bridge node:
cd visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/launch/bridgeonly.launch
./startkinect.sh

#start PD-Controller:
rosrun visual_mpc_rospkg joint_space_impedance.py


#start visual-mpc-client:
rosrun visual_mpc_rospkg visual_mpc_client.py <name_of_folder_inside:visual_mpc/experiments/cem_exp/benchmarks_sawyer>

#start visual-mpc-server (can be launched on newton4 or newton1):
rosrun visual_mpc_rospkg visual-mpc-server.py <name_of_folder_inside:visual_mpc/experiments/cem_exp/benchmarks_sawyer> --ngpu <number_of_gpus>



#Information for Simulation Setup:

In order to run rendering remotely on a different machine it is necessary to give remote users access to the x server:
Log into graphical session on remote computer and give access by typing:
xhost +

Also set the DISPLAY variable
export DISPLAY=:0

#To generate training data:
Go in the root directory of the repo and run:
python python/lsdc/parallel_data_collection.py  <name_of_datafolder>
Note: <name_of_datafolder> is the folder-name inside lsdc/pushing_data
Each of the folders in pushing_data must have a "/train" and "/test" subdirectory.
The hyperparams.py file to specifies how the data-collection is done.


#To run a benchmark on the pushing task go in the lsdc base directory and run:
python python/lsdc/utility/benchmarks.py <benchmark_folder_name>



