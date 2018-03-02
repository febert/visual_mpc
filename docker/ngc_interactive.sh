ngc batch run -n "Cloud-nv-us-west-2-111050" \
 -i "ucb_rail8888/tf1.4_gpu:based_nvidia" \
  --ace nv-us-west-2 \
  -in ngcv1 \
  --result /results \
   --datasetid 8350:/workspace/pushing_data \
   -c "/bin/sleep 3600"


