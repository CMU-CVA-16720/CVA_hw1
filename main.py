from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    ## Q1.1
    # Gaussian blurs the image
    # LoG picks up edges
    # d/dx picks up vertical edges
    # d/dy picks up horizontal edges
    # Multiple scales to vary sensitivity: small finds small features, big finds big features
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
#    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
#    img = img[:,:,0] # test grayscale pics
    filter_responses = visual_words.extract_filter_responses(opts, img)
#    util.display_filter_responses(opts, filter_responses)

    ## Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
#    img_path = '../figures/textons/gas_station.jpg'
    img_path = '../figures/textons/beach.jpg'
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    print("Dictionary shape: {}".format(dictionary.shape))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    # Forcing wordmap to certain values for testing
#    for i in range(0,10):
#    	wordmap[30*i:30*(i+1),:] = np.ones([30,wordmap.shape[1]])*i
#    util.visualize_wordmap(wordmap)

    ## Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    # n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
