import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

class FeatureExtractor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        # Load a pre-trained VGG16 model for feature extraction (without the top layers)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global Average Pooling layer
        self.model = Model(inputs=base_model.input, outputs=x)
        
        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

    def load_and_preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.image_size)  # Resize the image to match the input size of VGG16
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = preprocess_input(img)  # Preprocess the image for VGG16
        return img

    def extract_features_from_folder(self, folder_path):
        features = []
        labels = []
        
        # Iterate through '0' and '1' subfolders
        for label in ['0', '1']:
            label_folder = os.path.join(folder_path, label)
            if os.path.isdir(label_folder):
                for img_name in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, img_name)
                    
                    # Load and preprocess the image
                    img = self.load_and_preprocess_image(img_path)
                    
                    # Extract features from the image using the VGG16 model
                    feature = self.model.predict(img)
                    features.append(feature[0])  # Flatten the features
                    labels.append(int(label))  # Assign the corresponding label (0 or 1)
        
        # Convert features and labels to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        return features, labels

    def extract_features(self, train_dir, test_dir, val_dir):
        # Extract features for training, testing, and validation datasets
        train_features, train_labels = self.extract_features_from_folder(train_dir)
        test_features, test_labels = self.extract_features_from_folder(test_dir)
        val_features, val_labels = self.extract_features_from_folder(val_dir)
        
        # Save the features and labels into numpy arrays
        np.save('train_features.npy', train_features)
        np.save('train_labels.npy', train_labels)
        np.save('test_features.npy', test_features)
        np.save('test_labels.npy', test_labels)
        np.save('val_features.npy', val_features)
        np.save('val_labels.npy', val_labels)
        
        print(f"Extracted {train_features.shape[0]} training features and labels.")
        print(f"Extracted {test_features.shape[0]} testing features and labels.")
        print(f"Extracted {val_features.shape[0]} validation features and labels.")
        
        return (train_features, train_labels), (test_features, test_labels), (val_features, val_labels)

# Example usage
train_dir = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\train'
test_dir = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\test'
val_dir = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\valid'
extractor = FeatureExtractor()
(train_features, train_labels), (test_features, test_labels), (val_features, val_labels) = extractor.extract_features(train_dir, test_dir, val_dir)