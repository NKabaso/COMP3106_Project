import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

def loadData(datasetPath, training_annotations, testing_annotations):
    """
    Load in the data and split into training and testing sets.
    """
    # Load annotations
    train_annotations = helpers.loadAnnotations(training_annotations)
    test_annotations = helpers.loadAnnotations(testing_annotations)
    
    # Load images and labels
    X_train, y_train = helpers.loadImagesAndLabels(datasetPath, train_annotations)
    X_test, y_test = helpers.loadImagesAndLabels(datasetPath, test_annotations)
    
    return X_train, y_train, X_test, y_test
    
    
        