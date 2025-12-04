import cv2 # computer vision library
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
import FeatureExtraction
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from model import SafetyNet

#Weighted safety score
weights = {
            'crosswalk': 0.20,
            'signal': 0.25,
            'vehicles_in_way': 0.3,
            'pedestrians': 0.10,
            'obstacles': 0.05,
        }

class SafetyNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super(SafetyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # output: safe/unsafe
        )

    def forward(self, x):
        return self.net(x)
class SafetyDataset(Dataset):
    def __init__(self, labeled_images, feature_extractor):
        self.data = []
        self.labels = []

        for _, (image, mode, _, block) in labeled_images.items():
            features = feature_extractor.analyze(image)
            x = feature_extractor.features_to_tensor(features)

            # label logic (same as corrected version)
            safe = (mode in ['1', '2']) and (block == 'not_blocked')
            y = 0 if safe else 1

            self.data.append(x)
            self.labels.append(y)

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(train_dataset, epochs=50, lr=1e-3):
    model = SafetyNet()
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss:.4f}")

    return model

def evaluate(model, dataset):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for x, y in dataset:
            output = model(x)
            pred = torch.argmax(output).item()
            if pred == y:
                correct += 1

    acc = correct / len(dataset)
    print(f"Accuracy: {acc*100:.2f}%")



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
    
    # Create dataset
    train_dataset = SafetyDataset(labeled_images, feature_analyzer)

    # Train NN
    model = train_model(train_dataset, epochs=25)

    # Evaluate
    evaluate(model, train_dataset)

    # Save model
    torch.save(model.state_dict(), "safety_net.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()
    
    #single_image_test('dataset\heon_IMG_0556.JPG')
        
        
        

        
    
     