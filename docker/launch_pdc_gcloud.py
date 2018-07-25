import argparse
import os
import random


# format goes in order: proj_id, instance_num, nworkers, nsplit, isplit, exp_name
default_command = """
gcloud beta compute --project={} instances create-with-container instance-{} --zone=us-west1-b --machine-type=n1-standard-96 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=549992882073-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --min-cpu-platform=Intel\ Skylake --image=cos-stable-67-10575-62-0 --image-project=cos-cloud --boot-disk-size=500GB --boot-disk-type=pd-ssd --boot-disk-device-name=instance-1 --container-image=us.gcr.io/visualmpc-210823/mj_tf_cpu:latest --container-restart-policy=always --container-privileged --container-stdin --container-tty --container-command="/bin/bash" --container-arg="-c" --container-arg="cd visual_mpc; git checkout integrate_env; git pull origin integrate_env; python python_visual_mpc/visual_mpc_core/parallel_data_collection.py --nworkers {} --nsplit {} --isplit {} --cloud {}" --container-mount-host-path=mount-path=/result,host-path=/home/chronos/,mode=rw --labels=container-vm=cos-stable-67-10575-62-0
"""


def create_instance(proj_id, instance_num, exp_name, nworkers, nsplit):
    instance_tag = int(random.getrandbits(32))
    gcloud_command = default_command.format(proj_id, instance_tag, nworkers, nsplit, instance_num, exp_name)
    print('requesting instance: {}'.format(instance_tag))
    print(gcloud_command)
    ret_val = os.system(gcloud_command)
    print('returned with {}'.format(ret_val))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run parallel data collection on cloud')
    parser.add_argument('project_id', type=str, help="gcloud project id")
    parser.add_argument('experiment', type=str, help='experiment path (aka ../visual_mpc/<exp_path> should be valid hparams')
    parser.add_argument('ninstances', type=int, help='number of instances')
    parser.add_argument('--nworkers', type=int, help='number of worker proccesses per instance', default=120)
    args = parser.parse_args()

    for i in range(args.ninstances):
        create_instance(args.project_id, i, args.experiment, args.nworkers, args.ninstances)