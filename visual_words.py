import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from sklearn.cluster import KMeans


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

def compute_dictionary_one_image(img,opts):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    alpha = opts.alpha
    rand_rows = np.random.randint(0, img.shape[0],alpha)
    rand_cols = np.random.randint(0, img.shape[1],alpha)
    response = extract_filter_responses(opts,img)
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
    for ind,img_path in enumerate(train_files):
        # Prepend data directory
        img_path = data_dir+"/"+img_path
#        print("Dictionary {}/{}: {}".format(ind+1,len(train_files),img_path))
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        # helper fills filter_responses[ind*alpha:(ind+1)*alpha-1,:]
        filter_responses[ind*alpha:(ind+1)*alpha,:] = compute_dictionary_one_image(img,opts)
    # Compute k-means
    kmeans = KMeans(n_clusters=K,n_jobs=n_worker).fit(filter_responses)
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

