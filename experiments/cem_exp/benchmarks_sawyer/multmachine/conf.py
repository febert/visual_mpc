import os
current_dir = os.path.dirname(os.path.realpath(__file__))


# tf record data location:
LSDC_BASE = '/'.join(str.split(current_dir, '/')[:-4])

from python_visual_mpc.video_prediction.ray_setup_predictor import setup_predictor

configuration = {
'experiment_name': 'cem_control',
'setup_predictor': setup_predictor,
'current_dir': current_dir, #'directory for writing gifs' ,
# 'filepath of a pretrained model to use for cem
'pretrained_model': LSDC_BASE +'/tensorflow_data/sawyer/1stimg_bckgd_cdna/modeldata/model64002',
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'CDNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 10,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 200,           #batch size for evaluation' ,
'learning_rate': 0,     #'the base learning rate of the generator' ,
'file_visual': '',          # datafile used for making visualizations,
'kern_size': 9,              #size of DNA kerns
'sawyer':'',
'single_view':"",
'1stimg_bckgd':''
}