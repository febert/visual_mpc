import os
current_dir = os.path.dirname(os.path.realpath(__file__))

lsdc_home = '/'.join(str.split(current_dir, '/')[:-4])

from python_visual_mpc.video_prediction.setup_predictor_simple import setup_predictor

configuration = {
'experiment_name': 'cem_control',
'current_dir': current_dir, #'directory for writing gifs' ,
# 'filepath of a pretrained model to use for cem
'pretrained_model': lsdc_home + '/tensorflow_data/dna/modeldata/model46002',
'setup_predictor': setup_predictor,
'sequence_length': 15,      # 'sequence length, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 1,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 200,           #batch size for evaluation' ,
'learning_rate': 0,     #'the base learning rate of the generator' ,
'visualize': '',            #'load model from which to generate visualizations
'file_visual': '',          # datafile used for making visualizations,
'penal_last_only': False,
'kern_size':9,
}