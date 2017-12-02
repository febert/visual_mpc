import os
current_dir = os.path.dirname(os.path.realpath(__file__))


# tf record data location:
BASE = '/'.join(str.split(current_dir, '/')[:-4])

# from python_visual_mpc.video_prediction.setup_predictor_simple import setup_predictor
from python_visual_mpc.video_prediction.setup_predictor_towers import setup_predictor
from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import Base_Prediction_Model

configuration = {
'experiment_name': 'cem_control',
'pred_model': Base_Prediction_Model,
'setup_predictor': setup_predictor,
'current_dir': current_dir, #'directory for writing gifs' ,
# 'filepath of a pretrained model to use for cem
'pretrained_model': BASE + '/tensorflow_data/sawyer/weissgripper_basecls_20k/modeldata/model96002',
# 'pretrained_model': BASE + '/tensorflow_data/sawyer/weissgripper_basecls_20k/modeldata/model96002float16',
'sequence_length': 16,      # 'sequence length, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,                 #'Whether or not to give the state+action to the model' ,
'model': 'CDNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 10,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 200,           #batch size for evaluation' ,
'learning_rate': 0,     #'the base learning rate of the generator' ,
'file_visual': '',          # datafile used for making visualizations,
'kern_size': 9,              #size of DNA kerns
'sawyer':'',
'single_view':"",
'1stimg_bckgd':'',
'adim':5,
'sdim':4,
'ndesig':1,
'img_height':56,
'img_width':64,
'data_dir':'/pushing_data/weiss_gripper_20k/conf.py',
# 'float16':""
}
