import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.image as mpimg # for loading in images

def loadAnnotations(annotation_file, image_dir):
    """
    Load in the annotations from a CSV file
    """
    annotated_images = {}
    with open(annotation_file, 'r') as file:
        #dict_reader = csv.DictReader(file)
        annotations = csv.reader(file)
        headers = next(annotations)  # Read the header row
        max =500
        
        for entry in annotations:
            try:
                image_name = entry[0] #name of the image file
                mode = entry[1] #mode: '0' red, '1' green, '2' countdown green '3'countdown blank '4' None
                #midline coordinates of the pedestrian crossing based
                # coordinates are in pixels for a 4032x3024 image
                x1 = int(entry[2]) 
                y1 = int(entry[3])
                x2 = int(entry[4])
                y2 = int(entry[5])
                block = entry[6] #whether walkway is blocked
                
                # Look for either .jpg or .JPG
                if os.path.exists(f"{image_dir}/{image_name}"):
                    image_name = image_name
                elif os.path.exists(f"{image_dir}/{image_name.replace('.JPG', '.jpg')}"):
                    image_name = image_name.replace('.JPG', '.jpg')
                elif os.path.exists(f"{image_dir}/{image_name.replace('.jpg', '.JPG')}"):
                    image_name = image_name.replace('.jpg', '.JPG')
                
                image = mpimg.imread(f"{image_dir}/{image_name}")
                points = [x1/4032, y1/3024, x2/4032, y2/3024] #normalized coordinate values to be between [0, 1]
                #Images are stored as numpy arrays with shape (H = height, W= width, C = number of channels usually 3 for RGB)
                #PyTorch expects images to be in shape (C, H, W)
                #image = np.transpose(image, (2, 0, 1)) #change image shape from (H, W, C) to (C, H, W)
                points = torch.FloatTensor(points)

                annotated_images[image_name] = (image, mode, points, block) #store annotation info in dictionary
                if len(annotated_images) >= max:
                    break
            except Exception as e:
                print(f"Error: {e}")
                continue
            
    return annotated_images

def safety_one_hot_code(label):
    """
    One hot encode the label
    """    
    #Safe or not safe hot encoding
    #[safe, not safe]
    one_hot = np.zeros(2) #2 possible classes: safe, not safe
    light = label[1]
    block = label[3]
    if (light == '1' and block == 'not_blocked') or (light == '2' and block == 'not_blocked') or (label == "safe"): 
        one_hot[0] = 1 #safe to cross
    else:
        one_hot[1] = 1 #not safe to cross
    return one_hot


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

        