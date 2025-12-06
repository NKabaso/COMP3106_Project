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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import collections
#from model import SafetyNet


# Neural Network Classifier
class SafetyNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32): #8 input layers, 32 hidden layers
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
            features = feature_extractor.analyze(image) #extracts features
            x = feature_extractor.features_to_tensor(features) #converts to a tensor friendly array
             # UNSAFE = 1, SAFE = 0
            unsafe = False
            mode_str = str(mode).strip()
            block_str = str(block).strip().lower()
            # Rule 1: Red light (0) is ALWAYS unsafe
            if mode_str == '0':
                unsafe = True
            # Rule 2: Green (1) with blocked path is unsafe
            elif mode_str == '1' and block_str == 'blocked':
                unsafe = True
            # Rule 3: Countdown green (2) with blocked path is unsafe
            elif mode_str == '2' and block_str == 'blocked':
                unsafe = True
            # Rule 4: No signal (4) with blocked path is unsafe
            elif mode_str == '4' and block_str == 'blocked':
                unsafe = True
            # Everything else is safe
            else:
                unsafe = False
            
            y = 1 if unsafe else 0
            

            self.data.append(x)
            self.labels.append(y)

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(train_dataset, val_dataset, epochs=50, lr=1e-3):
    model = SafetyNet()
    #Dataset is converted in a feature vector and a label array
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        #Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                _, preds = torch.max(outputs, dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

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

def plot_confusion_matrix(model, dataset):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataset:
            logits = model(x)
            pred = torch.argmax(logits).item()

            all_preds.append(pred)
            all_labels.append(y)

    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1)
    recall = recall_score(all_labels, all_preds, pos_label=1)

    print(f"Precision (unsafe class): {precision:.4f}")
    print(f"Recall (unsafe class):    {recall:.4f}")
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Unsafe"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    
def single_image_test(model, image, feature_extractor):
    features = feature_extractor.analyze(image)
    x = torch.tensor(feature_extractor.features_to_tensor(features))
    logits = model(x)
    pred = torch.argmax(logits).item()
    if pred ==0:
        return "safe"
    else:
        return "unsafe"
    

def main():
    #Load in data
    #Directory paths
    #ImVisible: Pedestrian Traffic Light Dataset
    IMAGE_DIR = 'dataset'
    IMAGE_DIR_TEST = 'validation_testing_dataset'
    TRAINING_ANNOTATIONS = 'training_file.csv'
    VALIDATION_ANNOTATIONS = 'validation_file.csv'
    TESTING_ANNOTATIONS = 'testing_file.csv'
    
    labeled_training_images = helpers.loadAnnotations(TRAINING_ANNOTATIONS, IMAGE_DIR)
    labeled_validation_images = helpers.loadAnnotations(VALIDATION_ANNOTATIONS, IMAGE_DIR_TEST)
    label_testing_images = helpers.loadAnnotations(TESTING_ANNOTATIONS, IMAGE_DIR_TEST)
    #Create feature analyzer
    feature_extractor = FeatureExtraction.FeatureExtraction()
    
    # Create dataset
    train_dataset = SafetyDataset(labeled_training_images, feature_extractor)
    validation_dataset = SafetyDataset(labeled_validation_images, feature_extractor)
    testing_dataset = SafetyDataset(label_testing_images, feature_extractor)
    
    # Standardize features
    scaler = StandardScaler()
    scaler.fit(train_dataset.data)  # compute mean/std across ALL samples

    # apply scaling
    train_dataset.data = torch.tensor(scaler.transform(train_dataset.data), dtype=torch.float32)
    validation_dataset.data = torch.tensor(scaler.transform(validation_dataset.data), dtype=torch.float32)

    # Train NN
    model = train_model(train_dataset, validation_dataset, epochs=25)

    # Evaluate
    evaluate(model, validation_dataset)

    # Save model
    #torch.save(model.state_dict(), "safety_net.pth")
    
    plot_confusion_matrix(model, validation_dataset)
    
    #Random image test
    for cnt in range(5):
        random_key = random.choice(list(labeled_training_images.keys()))
        image, _, _, _ = labeled_training_images[random_key]
        output = single_image_test(model,image, feature_extractor)
        #print(f"It is {output} to cross")
        plt.imshow(image)
        h, w, _ = image.shape
        plt.text(w//2, 0, output, ha="center", color="black", fontsize=18) #labels image
        plt.show()
    
    

if __name__ == "__main__":
    main()
    
    #single_image_test('dataset\heon_IMG_0556.JPG')
        
        
        

        
    
     