from os.path import join
import os, multiprocessing
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import util
from opts import get_opts
import scipy.ndimage
import skimage.color
from sklearn.cluster import KMeans
from copy import copy

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # Create 3 channels for gray-scale images
    if len(img.shape) < 3:
        temp = np.zeros(img.shape[0:2] + (3,))
        for i in range(0,3):
            temp[:,:,i] = img
        img = temp
    # Massage image
    if(np.max(img)>1.0):
        img = img / np.max(img)
    img = skimage.color.rgb2lab(img)
    
    filter_scales = opts.filter_scales
    # Output will be W x H x Z
    # Z = 3 * 4 * len(opts.filter_scales)
    filter_responses = np.zeros(img.shape[0:2] + (3*4*len(opts.filter_scales),))
    # Loop through sizes
    mode = 'reflect'
    for ind,sigma in enumerate(opts.filter_scales):
        cursor = ind * 4 * 3
        # Gaussian
        filter_responses[:,:,cursor:cursor+3] = scipy.ndimage.gaussian_filter(img,(sigma,sigma,0),mode=mode)
        # LoG
        cursor += 3
        for j in range(0,3):
            filter_responses[:,:,cursor+j] = scipy.ndimage.gaussian_laplace(img[:,:,j],sigma,mode=mode)
        # dx
        cursor += 3
        filter_responses[:,:,cursor:cursor+3] = scipy.ndimage.gaussian_filter(img,(sigma,sigma,0),[0,1,0],mode=mode)
        # dy
        cursor += 3
        filter_responses[:,:,cursor:cursor+3] = scipy.ndimage.gaussian_filter(img,(sigma,sigma,0),[1,0,0],mode=mode)
    return filter_responses

def compute_dictionary_one_image(img_path,opts):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    # Set output file name
    out_name = img_path.replace('.','_').replace('/','_')
    # Get data and output directory
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    # Prepend data directory
    img_path = data_dir+"/"+img_path
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    alpha = opts.alpha
    # Use gaussian distribution for sampling
    rand_rows = (np.random.normal((img.shape[0]-1)/2, img.shape[0]/6,alpha)).astype(int)
    rand_cols = (np.random.normal((img.shape[1]-1)/2, img.shape[1]/6,alpha)).astype(int)
    # Make sure rows and cols are valid
    rand_rows[rand_rows >= img.shape[0]] = img.shape[0]-1
    rand_rows[rand_rows < 0] = 0
    rand_cols[rand_cols >= img.shape[1]] = img.shape[1]-1
    rand_cols[rand_cols < 0] = 0
    response = extract_filter_responses(opts,img)
    # Save response to file
    np.save(join(out_dir, out_name), response[rand_rows,rand_cols,:])
    return response[rand_rows,rand_cols,:]

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    # Seed for consistent random rows + cols, for repeatability to tune hyperparameters
    np.random.seed(0)

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
#    train_files = open(join(data_dir, 'train_files_lite.txt')).read().splitlines()
    # Create array to hold all filter responses (sample size * # sample images) x (3F)
    filter_responses = np.zeros([alpha * len(train_files), 3*4*len(opts.filter_scales)])
    # Prepare arg list
    arg_list = []
    for ind,img_path in enumerate(train_files):
        arg_list.append((img_path,opts))
    # Get filter responses
    with multiprocessing.Pool() as p:
        p.starmap(compute_dictionary_one_image, arg_list)
        p.close()
        p.join()
    # Read temporary files to matrix
    for ind,img_path in enumerate(train_files):
        file_name = img_path.replace('.','_').replace('/','_')+'.npy'
        temp_array = np.load(join(out_dir, file_name))
        filter_responses[ind*alpha:(ind+1)*alpha,:] = temp_array
        # Delete file after using
        os.remove(file_name)
    # Compute k-means
    kmeans = KMeans(n_clusters=K,n_jobs=n_worker,n_init=100).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
#    print("Dictionary size: {}".format(dictionary.shape))

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    wordmap = np.zeros(img.shape[0:2])
    # Get filter resp
    response = extract_filter_responses(opts,img)
    # Error matrix
    error = np.zeros([opts.K])
    # Loop each row
    for i,row in zip(range(0,img.shape[0]),response[:]):
        err_matrix = scipy.spatial.distance.cdist(row,dictionary)
        wordmap[i] = np.transpose(np.argmin(err_matrix,1))
    return wordmap


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
    * shape    : shape of image
    * layer    : current pyramid layer
    * indr    : row index of segment of interest
    * indc    : col index of segment of interest
    
    [output]
    * rng    : tuple of row and col slices
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
        # Calculate current layer without weight
        for r in range(current_layer.shape[0]):
            for c in range(current_layer.shape[1]):
                current_layer[r,c] = next_layer[2*r,2*c] + next_layer[2*r,2*c+1] + next_layer[2*r+1,2*c] + next_layer[2*r+1,2*c+1]
        # Apply weight
        if(l>0):
            current_layer /= 2
        # Save data
        hist_all = np.append(np.reshape(current_layer,2**(2*l)*K),hist_all)
#        print("sum({}x{}): {}".format(current_layer.shape[0],current_layer.shape[1],np.sum(current_layer)))
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
    wordmap = get_visual_words(opts, img, dictionary)
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
    # Prepare arg list
    arg_list = []
    for train_file in train_files:
        img_path = join(data_dir, train_file)
        arg_list.append((opts,img_path,dictionary))
    with multiprocessing.Pool() as p:
        proc_output = p.starmap(get_image_feature,arg_list)
        p.close()
        p.join()
    # Unpack output
    for i in range(0,len(proc_output)):
        features[i,:] = proc_output[i]
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
        # Check 10 minium distances
        tally = np.zeros(num_categories)
        for j in range(0,10):
            # Get index of minimum
            min_ind = np.argmin(distances)
            # Get classification of that index, update tally, set minimum to inf
            min_class = training_labels[min_ind]
            tally[min_class] += 1
            distances[min_ind] = float('inf')
        # Prediction is index of maximum in tally
        predicted = np.argmax(tally)
        actual = test_labels[i]
        # Update confusion matrix
        C[actual,predicted] += 1
        # Debug: print progress
#        print("Testing {}/{}, {}% ({}/{})".format(i+1,len(test_files),
#        100*np.trace(C)/np.sum(C),int(np.trace(C)),int(np.sum(C))))
    # Accuracy: tr(C)/sum(C)
    accuracy = np.trace(C)/np.sum(C)
    print("\nFinal: {}% ({}/{})".format(100*np.trace(C)/np.sum(C),
    int(np.trace(C)),int(np.sum(C))))
    
    return C, accuracy


def main():
    opts = get_opts()
    n_cpu = util.get_num_CPU()
    compute_dictionary(opts, n_worker=n_cpu)
    build_recognition_system(opts, n_worker=n_cpu)
    conf, accuracy = evaluate_recognition_system(opts, n_worker=n_cpu)
    print("Confustion matrix:\n{}".format(conf))
    print("Accuracy: {}%".format(accuracy*100))
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
