#ngc batch run -n "Cloud-nv-us-west-2-111050"\
#-i "ucb_rail8888/tf1.4_gpu:latest" \
#--ace nv-us-west-2 \
#-in ngcv1 \
#--command "/bin/sleep 3600" \
#--result /results \
#--datasetid 8350:/docker_home/pushing_data
ngc batch run -n "Cloud-nv-us-west-2-111050" \
 -i "ucb_rail8888/tf1.4_gpu:latest" \
  --ace nv-us-west-2 \
  -in ngcv1 \
  --result /results \
   --datasetid 8350:/docker_home/pushing_data,8522:/docker_home/pushing_data \
   -c "/bin/sleep 3600"


