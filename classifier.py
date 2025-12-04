import cv2 # computer vision library
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
import FeatureExtraction

#Weighted safety score
weights = {
            'crosswalk': 0.20,
            'signal': 0.25,
            'vehicles_in_way': 0.3,
            'pedestrians': 0.10,
            'obstacles': 0.05,
        }

def classifier(features):
    """Compute overall safety score based on extracted features."""
    safety_components ={}
    
    #Crosswalk
    if features['crosswalk_detected']:
        safety_components['crosswalk'] = features.get('crosswalk_confidence', 0.5)
    else:
        safety_components['crosswalk'] = 0.1
    
    #Traffic Light
    safety_components['traffic_light'] = features.get('signal_safety', 0.5)
    
    #Vehicles
    safety_components['vehicles_in_way'] = features.get('vehicle_present')
    if(safety_components['vehicles_in_way'] is None):
        safety_components['vehicles_in_way'] = 0.5
    else:
        safety_components['vehicles_in_way'] = 0
    
    # 4. Pedestrian occupancy
    pedestrians_in_crosswalk = features.get('pedestrians_in_crosswalk', 0)
    if pedestrians_in_crosswalk == 0:
        safety_components['pedestrians'] = 0.8  
    else: 0.3
    
    # 5. Obstacles
    obstacle_coverage = features.get('obstacle_coverage_ratio', 0)
    safety_components['obstacles'] = max(0.1, 1.0 - (obstacle_coverage * 5))  # Scale threat


    total_score = 0
    for component, weight in weights.items():
        total_score += safety_components.get(component, 0.5) * weight
    
    # Determine safety status
    if total_score >= 0.7:
        safety_status = 'safe'
    else:
        safety_status = 'unsafe'
        
    return total_score, safety_status
    
def single_image_test(image_path):
    #Load image
    image = mpimg.imread(image_path)
    #Image preprocessing if needed
    feature_extractor = FeatureExtraction.FeatureExtraction()
    features = feature_extractor.analyze(image)
    score, status = classifier(features)
    print(f"Image: {image_path}, Safety Score: {score:.2f}, Status: {status}")

def main():
    #Load in data
    #Directory paths
    #ImVisible: Pedestrian Traffic Light Dataset
    IMAGE_DIR = 'dataset'
    TRAINING_ANNOTATIONS = 'training_file.csv'
    TESTING_ANNOTATIONS = 'testing_file.csv'
    
    labeled_images = helpers.loadAnnotations(TRAINING_ANNOTATIONS, IMAGE_DIR)
    combined_labels = helpers.combine_labels_with_one_hot(labeled_images)
    
    #Create feature analyzer
    feature_analyzer = FeatureExtraction.FeatureExtraction()
    
    #predict labels
    predicted_labels = {}
    for image_name in labeled_images:
        image, label, _, _ = labeled_images[image_name]
        features = feature_analyzer.analyze(image)
        score, status = classifier(features)
        encoding = helpers.safety_one_hot_code(status)
        predicted_labels[image_name] = (encoding, score)
    
    #Evaluate model
    misclassified = 0
    total = len(combined_labels)
    for name, true_label in combined_labels.items():
        predicted_label, score = predicted_labels[name]
        if not np.array_equal(true_label, predicted_label):
            misclassified += 1
            
    accuracy = (total - misclassified) / total
    print(f'Accuracy: {accuracy*100:.2f}% ({total - misclassified}/{total})')
    print(f'Misclassified: {misclassified} out of {total}')


if __name__ == "__main__":
    main()
    
    #single_image_test('dataset\heon_IMG_0556.JPG')
        
        
        

        
    
     