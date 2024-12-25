from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("CancerBreast.csv")

# Separate features and labels
X = data.drop(columns="label", axis=1)
y = data['label']

# Ensure all features are numeric and handle missing values
X = X.select_dtypes(include=[np.number]).fillna(0)

# Ensure y is a 1D array
y = y.values.ravel() if hasattr(y, "values") else y

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

# 1. XGBoost with Hyperparameter Tuning
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
xgb_grid = GridSearchCV(xgb_model, xgb_params, scoring='accuracy', cv=3)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
xgb_acc = train_and_evaluate(best_xgb, X_train, y_train, X_test, y_test, name="XGBoost")

# 2. Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1)
svm_acc = train_and_evaluate(svm_model, X_train, y_train, X_test, y_test, name="SVM")

# 3. Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_acc = train_and_evaluate(dt_model, X_train, y_train, X_test, y_test, name="Decision Tree")

# 4. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf_acc = train_and_evaluate(rf_model, X_train, y_train, X_test, y_test, name="Random Forest")

# 5. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_acc = train_and_evaluate(knn_model, X_train, y_train, X_test, y_test, name="KNN")

# Summarize results
print("\nModel Performance Summary:")
print(f"XGBoost Accuracy: {xgb_acc}")
print(f"SVM Accuracy: {svm_acc}")
print(f"Decision Tree Accuracy: {dt_acc}")
print(f"Random Forest Accuracy: {rf_acc}")
print(f"KNN Accuracy: {knn_acc}")
