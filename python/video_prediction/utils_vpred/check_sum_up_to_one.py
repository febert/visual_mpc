import cPickle
import numpy as np

file_path = '/home/frederik/Documents/lsdc/experiments/cem_exp/data_files'

pix_distrib = cPickle.load(open(file_path + '/gen_distrib.pkl', "rb"))
gen_images = cPickle.load(open(file_path + '/gen_images.pkl', "rb"))
gtruth_images = cPickle.load(open(file_path + '/gtruth_images.pkl', "rb"))
concat_masks = cPickle.load(open(file_path + '/gen_masks.pkl', "rb"))

sequence_length = 15

# checking gen_distrib:
print 'summed values of pix distrib:'
for b in range(3):
    print 'batch index ', b
    for t in range(sequence_length - 1):
        print 'sum pixdistrib of tstep', t, ': ', np.sum(pix_distrib[t][b])


# checking gen_masks:
print 'summed values of masks:'

for b in range(3):
    print 'batch index ', b
    for t in range(sequence_length-1):
        pixel_sums =  np.sum(concat_masks[t][b], axis= 0)
        pixel_sums = np.sum(pixel_sums.squeeze())/64/64
        print 'pixel_sums', pixel_sums



