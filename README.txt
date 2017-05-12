
#Information for Setup:

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



