import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

def hsv(image):
    """
    Extract features from the image using color histograms
    """
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Compute color histograms for each channel
    #Arguments: image, channels, mask, histSize, ranges
    # CHANNELS: 0 - Hue, 1 - Saturation(intensity), 2 - Value(brightness)
    #Mask = None means it computes histogram for the whole image
    h_hist = cv2.calcHist([hsv_image], [0], None, [32], [0, 180]) 
    s_hist = cv2.calcHist([hsv_image], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [32], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Concatenate histograms into a single feature vector
    feature_vector = np.concatenate((h_hist, s_hist, v_hist))
    
    return feature_vector

      