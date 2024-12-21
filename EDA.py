import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')
val_features = np.load('val_features.npy')
val_labels = np.load('val_labels.npy')

features = np.concatenate((train_features, test_features, val_features), axis=0)
labels = np.concatenate((train_labels, test_labels, val_labels), axis=0)


data = pd.DataFrame(features)
data['label'] = labels


print(data.describe())

print(data.info)
print('null value\n',data.isnull().sum())

sns.countplot(x='label', data=data)
plt.title('Distribution of Labels')
plt.show()

sns.pairplot(data.iloc[:, :5], hue='label')
plt.show()

corr_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()