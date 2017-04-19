Information for Setup:

In order to run rendering remotely on a different machine it is necessary to give remote users access to the x server:
Log into graphical session on remote computer and give access by typing:
xhost +

Also set the DISPLAY variable
export DISPLAY=:0
