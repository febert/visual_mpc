# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from python_visual_mpc.region_proposal_networks.rpn_net.util.faster_rcnn_lib.fast_rcnn.config import cfg
from python_visual_mpc.region_proposal_networks.rpn_net.util.faster_rcnn_lib.nms.py_cpu_nms import py_cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return py_cpu_nms(dets, thresh)

    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)
