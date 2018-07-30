import os
import tensorflow as tf

def get_checkpoint_restore_saver(checkpoint, skip_global_step=False, restore_to_checkpoint_mapping=None,
                                 restore_scope=None):
    if os.path.isdir(checkpoint):
        # latest_checkpoint doesn't work when the path has special characters
        checkpoint = tf.train.latest_checkpoint(checkpoint)
    checkpoint_reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint)
    checkpoint_var_names = checkpoint_reader.get_variable_to_shape_map().keys()
    restore_to_checkpoint_mapping = restore_to_checkpoint_mapping or (lambda name: name.split(':')[0])
    restore_vars = {restore_to_checkpoint_mapping(var.name): var
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=restore_scope)}
    if skip_global_step and 'global_step' in restore_vars:
        del restore_vars['global_step']
    # restore variables that are both in the global graph and in the checkpoint
    restore_and_checkpoint_vars = {name: var for name, var in restore_vars.items() if name in checkpoint_var_names}
    restore_saver = tf.train.Saver(max_to_keep=1, var_list=restore_and_checkpoint_vars, filename=checkpoint)
    # print out information regarding variables that were not restored or used for restoring
    restore_not_in_checkpoint_vars = {name: var for name, var in restore_vars.items() if
                                      name not in checkpoint_var_names}
    checkpoint_not_in_restore_var_names = [name for name in checkpoint_var_names if name not in restore_vars]
    if skip_global_step and 'global_step' in checkpoint_not_in_restore_var_names:
        checkpoint_not_in_restore_var_names.remove('global_step')
    if restore_not_in_checkpoint_vars:
        print("global variables that were not restored because they are "
              "not in the checkpoint:")
        for name, _ in sorted(restore_not_in_checkpoint_vars.items()):
            print("    ", name)
    if checkpoint_not_in_restore_var_names:
        print("checkpoint variables that were not used for restoring "
              "because they are not in the graph:")
        for name in sorted(checkpoint_not_in_restore_var_names):
            print("    ", name)
    return restore_saver, checkpoint
