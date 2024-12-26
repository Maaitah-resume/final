import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import load_img, img_to_array 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def loadImages(dir, target_size=(128, 128)):
    X = []
    y = []
    for label in ['0', '1']:
        class_dir = os.path.join(dir, label)
        if os.path.isdir(class_dir):
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                try:
                    img = load_img(img_path, target_size=target_size)
                    imgArray = img_to_array(img)
                    X.append(imgArray)
                    y.append(int(label))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(X), np.array(y)
    
trainDir = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\train'
validDir = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\valid'
testDir = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\test'

X_train, y_train = loadImages(trainDir)
X_valid, y_valid = loadImages(validDir)
X_test, y_test = loadImages(testDir)

X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_valid_flat = X_valid.reshape(X_valid.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier()
}

param_grids = {
    'KNeighborsClassifier': {
        'n_neighbors': randint(3, 10),
        'weights': ['uniform', 'distance'],
    },
    'SVC': {
        'C': [0.1, 1, 5],
        'kernel': ['linear', 'rbf'],
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
       
    },
    'RandomForestClassifier': {
        'n_estimators': randint(100, 300),
        'max_depth': [None, 10, 20, 30],
       
    }
}
for name, model in models.items():
    print(f"Hyper-tuning {name}...")
    random_search = RandomizedSearchCV(model, param_grids[name], cv=3, scoring='accuracy', verbose=1, n_jobs=-1, n_iter=10)
    random_search.fit(X_train_flat, y_train)
    best_model = random_search.best_estimator_
    print(f"Best parameters for {name}: {random_search.best_params_}")
    
    y_pred_valid = best_model.predict(X_valid_flat)
    print(f"Validation report for {name}:\n")
    print(classification_report(y_valid, y_pred_valid))
    
    y_pred_test = best_model.predict(X_test_flat)
    print(f"Test report for {name}:\n")
    print(classification_report(y_test, y_pred_test))