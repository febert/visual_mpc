filename = "/home/frederik/Dokumente/lsdc/experiments/lsdc_exp/data_files/tfrecords/traj_no0.tfrecords"
# filename = "/tmp/data/train.tfrecords"
print filename
import tensorflow as tf

for serialized_example in tf.python_io.tf_record_iterator(filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # label = example.features.feature['label']
    #
    # print label


    for index in range(30):
        # traverse the Example format to get data
        action = example.features.feature['move/' + str(index) + '/action']
        state = example.features.feature['move/' + str(index) + '/state']
        image = example.features.feature['move/' + str(index) + '/image/encoded']

        # print state
        print action
        # print image
