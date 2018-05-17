# argument 1: name of job
ngc batch list -d 240h | grep 'Frederik Ebert' | grep '$1' | awk '{print $2}' | xargs -n 1 -P 5 ngc result remove