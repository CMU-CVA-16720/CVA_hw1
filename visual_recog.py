import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    to_rtn = np.histogram(wordmap,range(0,K+1))[0]
    return (to_rtn/np.sum(to_rtn))

def split_ind(shape,layer,indr,indc):
    '''
    Splits total into several parts, returning range for one part
    
    [input]
    * shape	: shape of image
    * layer	: current pyramid layer
    * indr	: row index of segment of interest
    * indc	: col index of segment of interest
    
    [output]
    * rng	: tuple of row and col slices
    '''
    row_rng = slice(shape[0]*indr//(2**layer),shape[0]*(indr+1)//(2**layer))
    col_rng = slice(shape[1]*indc//(2**layer),shape[1]*(indc+1)//(2**layer))
    return (row_rng,col_rng)

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # Calculate last layer first
    last_layer_hist = np.array([]) # top left -> top right -> bottom left -> bottom right
    for r in range(0,2**L):
    	for c in range(0,2**L):
    		wordmap_segment = wordmap[split_ind(wordmap.shape,L,r,c)]
	    	last_layer_hist = np.append(last_layer_hist,get_feature_from_wordmap(opts,wordmap_segment))
    # Normalize & weigh last layer hist
    last_layer_hist /= np.sum(last_layer_hist)
    if(L>0):
    	last_layer_hist /= 2
    # Calculate all other layers
    next_layer = np.reshape(last_layer_hist,[2**L,2**L,K])
    hist_all = last_layer_hist
#    print("last_layer sum: {}".format(np.sum(next_layer)))
    for l in reversed(range(0,L)):
    	current_layer = np.zeros([2**l,2**l,K])
    	# Calculate current layer with weight
    	for r in range(current_layer.shape[0]):
    		for c in range(current_layer.shape[1]):
    			current_layer[r,c] = next_layer[2*r,2*c] + next_layer[2*r,2*c+1] + next_layer[2*r+1,2*c] + next_layer[2*r+1,2*c+1]
    	# Apply weight
    	if(l>0):
    		current_layer /= 2
    	# Save data
    	hist_all = np.append(np.reshape(current_layer,2**(2*l)*K),hist_all)
#    	print("sum({}x{}): {}".format(current_layer.shape[0],current_layer.shape[1],np.sum(current_layer)))
    	# update next layer
    	next_layer = current_layer
#    print("Final hist shape, sum: {}; {}".format(hist_all.shape,np.sum(hist_all)))
    return hist_all

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    # Retrieve image
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    # Get wordmap
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # Get SPM histogram for image
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
#    train_files = open(join(data_dir, 'train_files_lite.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
#    train_labels = np.loadtxt(join(data_dir, 'train_labels_lite.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    # Get features for training data
    features = np.zeros([len(train_files),int(opts.K*(4**(opts.L+1)-1)/3)])
    for i,train_file in enumerate(train_files):
    	# Debug: print progress
    	print("Training {}/{}".format(i+1,len(train_files)))
    	# Get image path, then compute feature
    	img_path = join(opts.data_dir, train_file)
    	features[i,:] = get_image_feature(opts,img_path,dictionary)
    # Get SPM layer number
    SPM_layer_num = opts.L+1

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    hist_dist = 1-np.sum(np.minimum(word_hist,histograms),axis=1)
    return hist_dist
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    training_features = trained_system['features']
    training_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    
    # Create confusion matrix: C(i,j) = count of class i being predicted as class j
    # 8 categories: aquarium, desert, highway, kitchen, laundromat, park, waterfall, windmill
    num_categories = 8
    C = np.zeros([num_categories, num_categories])
    
    # Iterate through test images, make prediction, update C
    for i,test_file in enumerate(test_files):
    	# Get image and its feature
    	img_path = join(opts.data_dir, test_file)
    	feature = get_image_feature(opts,img_path,dictionary)
    	# Get prediction and actual
    	distances = distance_to_set(feature, training_features)
    	predicted = training_labels[np.argmin(distances)]
    	actual = test_labels[i]
    	# Update confusion matrix
    	C[actual,predicted] += 1
    	# Debug: print progress
    	print("Testing {}/{}, {}% ({}/{})".format(i+1,len(test_files),
    	100*np.trace(C)/np.sum(C),int(np.trace(C)),int(np.sum(C))))
    # Accuracy: tr(C)/sum(C)
    accuracy = np.trace(C)/np.sum(C)
    print("\nFinal: {}% ({}/{})".format(100*np.trace(C)/np.sum(C),
    int(np.trace(C)),int(np.sum(C))))
    
    return C, accuracy
    


