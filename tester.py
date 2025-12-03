
from helper import helpers
IMAGE_DIR = 'dataset'
TRAINING_ANNOTATIONS = 'training_annotations.csv'
TESTING_ANNOTATIONS = 'testing_annotations.csv'

def test_labels():
    labeled_images = loadAnnotations(TRAINING_ANNOTATIONS, IMAGE_DIR)
    for name, label in labeled_images.items():
        one = one_hot_code(label)
        print(f"Image: {name}, Label: {label}, One-Hot: {one}")
        

if __name__ == "__main__":
    test_labels()
        