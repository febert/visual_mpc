import tensorflow as tf
import argparse
import importlib.machinery
import importlib.util
from python_visual_mpc.goal_classifier.models.base_model import BaseGoalClassifier
from python_visual_mpc.goal_classifier.datasets.base_classifier_dataset import ClassifierDataset
import os


def train(conf, batch_paths, restore_path, device, save_freq, summary_freq, tboard_port):
    print('Using GPU: {}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    dataset_type, dataset_params = conf.pop('dataset', ClassifierDataset), conf.pop('dataset_params', {})
    datasets = [dataset_type(d[0], d[1], dataset_params) for d in batch_paths]

    model = conf.pop('model', BaseGoalClassifier)(conf, datasets)
    global_step = tf.train.get_or_create_global_step()
    model.build(global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model.restore(sess, restore_path)
    summary_writer = tf.summary.FileWriter(model.save_dir, graph=sess.graph, flush_secs=10)
    if tboard_port > 0:
        os.system("tensorboard --logdir {}  --port {} &".format(model.save_dir, tboard_port))

    start_step = global_step.eval(sess)
    for _ in range(model.max_steps - start_step):
        i_step = global_step.eval(sess)
        sess_fetches = model.step(sess, global_step, eval_summaries=i_step % summary_freq == 0)
        if i_step % summary_freq == 0:
            print('iter {} - train loss: {}, '
                  'val loss: {}'.format(i_step, sess_fetches['train_loss'], sess_fetches['val_loss']), end="\r")

            itr_summary = tf.Summary()
            for f in sess_fetches.keys():
                if 'summary' in f:
                    itr_summary.value.add(tag=f, simple_value=sess_fetches[f])
            summary_writer.add_summary(itr_summary, i_step)
            print()
        else:
            print('iter {} - train loss: {}'.format(i_step, sess_fetches['train_loss']), end="\r")

        if i_step > 0 and i_step % save_freq == 0:
            model.save(sess, i_step)
    model.save(sess, global_step.eval(sess))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path', type=str, help="Path to the classifier conf file")
    parser.add_argument("--input_dirs", "--input_dir", type=str, nargs='+', required=True,
                        help="either a directory containing subdirectories train, val, test which contain records")
    parser.add_argument("--train_batch_sizes", type=int, nargs='+', help="splits for the training datasets",
                        required=True)
    parser.add_argument('--restore_path', type=str, default=None, help="path to existing model file")
    parser.add_argument('--device', type=int, default=0, help= "GPU use for training")
    parser.add_argument('--save_freq', type=int, default=5000, help="Checkpoint Save Frequency")
    parser.add_argument('--summary_freq', type=int, default=50, help="Summary Save Frequency")
    parser.add_argument('--port', type=int, default=-1, help="Open a summary tensorboard to this port")
    args = parser.parse_args()

    loader = importlib.machinery.SourceFileLoader('classifier_conf', args.conf_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    conf = mod.config

    train(conf, list(zip(args.input_dirs, args.train_batch_sizes)),
          args.restore_path, args.device, args.save_freq, args.summary_freq, args.port)
