import os

job_start_id = 41212
n = 10
for j in range(n):
    cmd = "ngc batch kill {} &".format(job_start_id + j)
    print(cmd)
    os.system(cmd)
