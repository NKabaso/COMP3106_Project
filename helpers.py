import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.image as mpimg # for loading in images

def loadData():
    """
    Load in the data
    """
    #Directory paths
    #ImVisible: Pedestrian Traffic Light Dataset
    IMAGE_DIR = 'dataset'
    TRAINING_ANNOTATIONS = 'training_annotations.csv'
    TESTING_ANNOTATIONS = 'testing_annotations.csv'
    
    #Dictionaries
    labeled_images = {}
    
    # Load annotations
    with open(TRAINING_ANNOTATIONS, 'r') as file:
        dict_reader = csv.DictReader(file)
        
        for entry in dict_reader:
            image_name = entry['ï»¿file'] #name of the image file
            image_name= image_name.replace('.JPG', '.jpg') #ensure file extension is correct
            mode = entry['mode'] #mode: '0' red, '1' green, '2' countdown green '3'countdown blank '4' None
            #midline coordinates of the pedestrian crossing based
            # coordinates are in pixels for a 4032x3024 image
            x1 = int(entry['x1']) 
            y1 = int(entry['y1'])
            x2 = int(entry['x2'])
            y2 = int(entry['y2'])
            block = entry['block'] #whether walkway is blocked
            
            image = mpimg.imread(f"{IMAGE_DIR}/{image_name}")
            points = [x1/4032, y1/3024, x2/4032, y2/3024] #normalized coordinate values to be between [0, 1]
            #Images are stored as numpy arrays with shape (H = height, W= width, C = number of channels usually 3 for RGB)
            #PyTorch expects images to be in shape (C, H, W)
            image = np.transpose(image, (2, 0, 1)) #change image shape from (H, W, C) to (C, H, W)
            points = torch.FloatTensor(points)
            labeled_images[image_name] = (image, mode, points, block) #store image and its label info in dictionary
            
    return labeled_images

def loadAnnotations(annotation_file, image_dir):
    """
    Load in the annotations from a CSV file
    """
    annotated_images = {}
    with open(annotation_file, 'r') as file:
        dict_reader = csv.DictReader(file)
        max =500
        
        for entry in dict_reader:
            try:
                image_name = entry['ï»¿file'] #name of the image file
                image_name= image_name.replace('.JPG', '.jpg') #ensure file extension is correct
                mode = entry['mode'] #mode: '0' red, '1' green, '2' countdown green '3'countdown blank '4' None
                #midline coordinates of the pedestrian crossing based
                # coordinates are in pixels for a 4032x3024 image
                x1 = int(entry['x1']) 
                y1 = int(entry['y1'])
                x2 = int(entry['x2'])
                y2 = int(entry['y2'])
                block = entry['block'] #whether walkway is blocked
                image = mpimg.imread(f"{image_dir}/{image_name}")
                print(f"Loaded image: {image.shape}")
                points = [x1/4032, y1/3024, x2/4032, y2/3024] #normalized coordinate values to be between [0, 1]
                #Images are stored as numpy arrays with shape (H = height, W= width, C = number of channels usually 3 for RGB)
                #PyTorch expects images to be in shape (C, H, W)
                #image = np.transpose(image, (2, 0, 1)) #change image shape from (H, W, C) to (C, H, W)
                points = torch.FloatTensor(points)

                annotated_images[image_name] = (image, mode, points, block) #store annotation info in dictionary
                if len(annotated_images) >= max:
                    break
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
                continue
            
    return annotated_images

def safety_one_hot_code(label):
    """
    One hot encode the label
    """    
    #Safe or not safe hot encoding
    #[safe, not safe]
    one_hot = np.zeros(2) #2 possible classes: safe, not safe
    if (label[1] == '1' and label[3] == 'not_blocked') or (label[1] == '2' and label[3] == 'not_blocked') or (label == "safe"): 
        one_hot[0] = 1 #safe to cross
    else:
        one_hot[1] = 1 #not safe to cross
    return one_hot

def one_hot_code(label):
    """
    One hot encode the label
    """
    one_hot = np.zeros(6) #5 possible classes: red, green, countdown green, countdown blank, none
    if label[1] == '0':
        one_hot[0] = 1 #red
    if label[1] == '1':
        one_hot[1] = 1 #green
    if label[1] == '2':
        one_hot[2] = 1 #countdown green
    if label[1] == '3':
        one_hot[3] = 1 #countdown blank
    if label[1] == '4':
        one_hot[4] = 1 #none
    if label[3] == 'blocked':
        one_hot[5] = 1 #blocked
    return one_hot

def combine_labels_with_one_hot(labeled_images):
    """
    Combine the one hot encoded labels with the labeled images
    """
    combined_data = {}
    #for image_name, (image, mode, points, block) in labeled_images.items():
     #   
      #  one_hot_label = one_hot_code((image_name, mode, mode, mode, block))
      #  combined_data[image_name] = (image, one_hot_label, points, block)
    for image_name, label in labeled_images.items():  
        #one_hot_label = one_hot_code(label)
        one_hot_label = safety_one_hot_code(label)
        combined_data[image_name] = one_hot_label # image name mapped to one hot label
    return combined_data


def test_labels():
    IMAGE_DIR = 'dataset'
    TRAINING_ANNOTATIONS = 'sample.csv'
    TESTING_ANNOTATIONS = 'testing_file.csv'
    
    labeled_images = loadAnnotations(TRAINING_ANNOTATIONS, IMAGE_DIR)
    for name, label in labeled_images.items():
        one = safety_one_hot_code(label)
        print(f"Image: {name} One-Hot: {one}")
        plt.imshow(label[0])
        plt.show()
        

if __name__ == "__main__":
    test_labels()

        