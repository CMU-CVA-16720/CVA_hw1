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
    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    ## Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    # Comment/uncomment img_path as desired
    img_path = '../data/aquarium/sun_acdxxtklhqaoxnqd.jpg'
    img_path = '../data/park/labelme_gmgfmbtnkpwaugo.jpg'
    img_path = '../data/desert/sun_adpbjcrpyetqykvt.jpg'
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
#    util.visualize_wordmap(wordmap)

    ## Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    print("Confusion matrix:\n{}".format(conf))
    print("Accuracy: {}%".format(100*accuracy))
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

    util.visualize_wordmap(wordmap)


if __name__ == '__main__':
    main()
