import tensorflow as tf
import argparse
import importlib.machinery
import importlib.util
from python_visual_mpc.goal_classifier.models.base_model import BaseGoalClassifier
from python_visual_mpc.visual_mpc_core.Datasets.classifier_dataset import ClassifierDataset


def train(conf, batch_paths, restore_path):
    dataset_type, dataset_params = conf.pop('dataset', ClassifierDataset), conf.pop('dataset_params', {})
    datasets = [dataset_type(d[0], d[1], dataset_params) for d in batch_paths]

    model = conf.pop('model', BaseGoalClassifier)(conf, datasets)
    global_step = tf.train.get_or_create_global_step()
    model.build()

    print(1 / 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path', type=str, help="Path to the classifier conf file")
    parser.add_argument("--input_dirs", "--input_dir", type=str, nargs='+', required=True,
                        help="either a directory containing subdirectories train, val, test which contain records")
    parser.add_argument("--train_batch_sizes", type=int, nargs='+', help="splits for the training datasets",
                        required=True)
    parser.add_argument('--restore_path', type=str, default=None)
    args = parser.parse_args()

    loader = importlib.machinery.SourceFileLoader('classifier_conf', args.conf_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    conf = mod.config

    train(conf, list(zip(args.input_dirs, args.train_batch_sizes)), args.restore_path)
