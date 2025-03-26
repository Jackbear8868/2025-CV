# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_size = 16
    tiny_img_feats = []

    for img_path in img_paths:
        # 1. Open
        img = Image.open(img_path)
        
        # 2. Crop center square
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

        # 3. Resize to tiny image
        img = img.resize((tiny_size, tiny_size), Image.Resampling.LANCZOS)

        # 4. Convert to numpy, flatten, and normalize
        vec = np.array(img, dtype=np.float32).flatten()

        # Normalize: zero mean and unit length
        vec -= np.mean(vec)
        if np.linalg.norm(vec) > 0:
            vec /= np.linalg.norm(vec)

        tiny_img_feats.append(vec)

    tiny_img_feats = np.vstack(tiny_img_feats)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################
    
    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 800
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################

    all_sift_descriptors = []

    num = 100
    step_size = 6

    for img_path in img_paths:
        img = Image.open(img_path)
        img_np = np.array(img, dtype=np.float32)

        # Use larger step size and fast=True for speed
        _, descriptors = dsift(img_np, step=[step_size, step_size], fast=True)

        if descriptors.shape[0] == 0:
            continue  # skip image if no SIFT features found

        # Optionally sample descriptors to reduce computation
        if descriptors.shape[0] > num:
            idx = np.random.choice(descriptors.shape[0], num, replace=False)
            descriptors = descriptors[idx]

        all_sift_descriptors.append(descriptors.astype(np.float32))


    all_sift_descriptors = np.vstack(all_sift_descriptors)

    vocab = kmeans(all_sift_descriptors, num_centers=vocab_size)

    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    vocab_size = vocab.shape[0]

    for img_path in img_paths:
        img = Image.open(img_path)
        img_np = np.array(img, dtype=np.float32)

        # Use larger step size and fast=True for speed
        _, descriptors = dsift(img_np, step=[7, 7], fast=True)

        if descriptors.shape[0] == 0:
            # If no features found, use zero histogram
            hist = np.zeros(vocab_size)
        else:
            # Compute distances
            distances = cdist(descriptors, vocab, metric='euclidean')

            # Soft assignment: use top-k nearest visual words
            topk = 3
            hist = np.zeros(vocab_size, dtype=np.float32)

            nearest_indices = np.argsort(distances, axis=1)[:, :topk]
            nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

            weights = np.exp(-nearest_distances)
            weights /= np.sum(weights, axis=1, keepdims=True)

            for i in range(nearest_indices.shape[0]):
                for j in range(topk):
                    hist[nearest_indices[i, j]] += weights[i, j]

            # Normalize histogram (RootSIFT)
            hist = np.sqrt(hist)
            hist /= np.linalg.norm(hist) + 1e-8

        img_feats.append(hist)

    img_feats = np.vstack(img_feats)

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    k = 8  # Number of neighbors to consider

    # Step 0: Convert training labels to numeric IDs
    train_label_ids = np.array([CAT2ID[label] for label in train_labels])
    
    # Step 1: Compute distances
    distances = cdist(test_img_feats, train_img_feats, metric='cosine')

    # Step 2: 取前 k 個最相近的 training feature
    nearest_indices = np.argsort(distances, axis=1)[:, :k]

    # Step 3: 加權投票
    test_predicts = []

    for i in range(len(test_img_feats)):
        neighbor_ids = train_label_ids[nearest_indices[i]]
        neighbor_distances = distances[i, nearest_indices[i]]

        # 轉為權重（距離越小權重越大），防止除以 0 加 1e-8
        weights = 1.0 / (neighbor_distances + 1e-8)

        # 加權投票
        votes = np.bincount(neighbor_ids, weights=weights, minlength=len(CAT))
        predicted_id = np.argmax(votes)
        predicted_label = CAT[predicted_id]
        test_predicts.append(predicted_label)


    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts
