import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

# Directories for training, validation, and test data
trainDir0 = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\train\\0'
trainDir1 = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\train\\1'
validDir0 = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\valid\\0'
validDir1 = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\valid\\1'
testDir0 = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\test\\0'
testDir1 = 'C:\\Users\\maait\\Desktop\\University\\Machine learning\\final\\test\\1'

x_train = []
y_train = []
x_test = []
y_test = []
x_valid = []
y_valid = []

for files in os.listdir(trainDir0):
    x_train.append(os.path.join(trainDir0, files))
    y_train.append(0)
for files in os.listdir(trainDir1):
    x_train.append(os.path.join(trainDir1, files))
    y_train.append(1)
for files in os.listdir(testDir0):
    x_test.append(os.path.join(testDir0, files))
    y_test.append(0)
for files in os.listdir(testDir1):
    x_test.append(os.path.join(testDir1, files))
    y_test.append(1)
for files in os.listdir(validDir0):
    x_valid.append(os.path.join(validDir0, files))
    y_valid.append(0)
for files in os.listdir(validDir1):
    x_valid.append(os.path.join(validDir1, files))
    y_valid.append(1)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
print(len(x_valid))
print(len(y_valid))

num_0_train = 0
num_1_train = 0
num_0_test = 0
num_1_test = 0
num_0_valid = 0
num_1_valid = 0

for label in y_train:
    if label == 0:
        num_0_train += 1
    else:
        num_1_train += 1
for label in y_test:
    if label == 0:
        num_0_test += 1
    else:
        num_1_test += 1
for label in y_valid:
    if label == 0:
        num_0_valid += 1
    else:
        num_1_valid += 1

print(num_0_train)
print(num_1_train)
print(num_0_test)
print(num_1_test)
print(num_0_valid)
print(num_1_valid)

x_train_images = []
x_test_images = []
x_valid_images = []

for i in range(len(x_train)):
    img = cv2.imread(x_train[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    x_train_images.append(img)
for i in range(len(x_test)):
    img = cv2.imread(x_test[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    x_test_images.append(img)
for i in range(len(x_valid)):
    img = cv2.imread(x_valid[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    x_valid_images.append(img)

x_train_images_np = np.array(x_train_images)
x_test_images_np = np.array(x_test_images)
x_valid_images_np = np.array(x_valid_images)


y_train_np = np.array(y_train)
y_test_np = np.array(y_test)
y_valid_np = np.array(y_valid)


data = np.concatenate([x_train_images_np, x_test_images_np, x_valid_images_np])
labels = np.concatenate([y_train_np, y_test_np, y_valid_np])
print(data.shape)
print(labels.shape)

data, labels = shuffle(data, labels)

image = data[0]
if image is not None:
    print("Image dimensions:", image.shape)

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
else:
    print("Error")

x_train_flat = x_train_images_np.reshape(x_train_images_np.shape[0], -1)
x_test_flat = x_test_images_np.reshape(x_test_images_np.shape[0], -1)
x_valid_flat = x_valid_images_np.reshape(x_valid_images_np.shape[0], -1)

models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}

params = { 
    "KNeighborsClassifier": {"n_neighbors": [3, 5, 7, 9, 11]},
    "LogisticRegression":    {"C": [0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"]},  
    "DecisionTreeClassifier": {"max_depth": [3, 5, 7, 9, 11], "min_samples_split": [2, 4, 6, 8, 10], "min_samples_leaf": [1, 2, 3, 4, 5]},
    "RandomForestClassifier": {"n_estimators": [100, 200, 300, 400, 500], "max_depth": [3, 5, 7, 9, 11], "min_samples_split": [2, 4, 6, 8, 10], "min_samples_leaf": [1, 2, 3, 4, 5]}
}

for key in models:
    model = models[key]
    param = params[key]
    grid = GridSearchCV(model, param, cv=5)
    grid.fit(x_train_flat, y_train_np)
    print(key)
    print(grid.best_params_)
    print(grid.best_score_)
    print(grid.best_estimator_)
    print("\n")