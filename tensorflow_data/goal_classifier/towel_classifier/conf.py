from python_visual_mpc.goal_classifier.datasets.infer_label_dataset import InferLabelDataset
from python_visual_mpc.goal_classifier.models.unconditioned_model import UnconditionedGoalClassifier


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
config = {
    'dataset': InferLabelDataset,
    'model': UnconditionedGoalClassifier,
    'save_dir': BASE_DIR + '/base_model_small/',
    'lr': 1e-5,
    'num_channels': [32, 16],
    'fc_layers': [32,16]
}
