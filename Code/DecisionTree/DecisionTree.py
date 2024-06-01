
# Loading, Preprocessing, and Splitting the Dataset with Reducing Image
# Decision Tree supervised

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns

# Load Images Function
def load_images(base_path, classes, image_size):
    images = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(base_path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(cls)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Define paths and classes
base_path = "D:/Concordia/Applied AI/Summer 2024/Project - Place 365/COMP6721"  # Update this with the correct path
classes = ["Airport_terminal", "Market", "Movie_theater", "Museum", "Restaurant"]
image_size = (64, 64)  # Reduced image size

# Load images and labels
images, labels = load_images(base_path, classes, image_size)

# Encode labels as integers
label_to_index = {label: index for index, label in enumerate(classes)}
encoded_labels = np.array([label_to_index[label] for label in labels])

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, encoded_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert images to float32 and normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data for Decision Tree
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_reshaped, y_train)

# Predict on Validation and Test sets
y_val_pred = dt_classifier.predict(X_val_reshaped)
y_test_pred = dt_classifier.predict(X_test_reshaped)

# Calculate Metrics
def print_metrics(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"{dataset_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))
    return accuracy, precision, recall, f1

# Print Metrics for Validation and Test sets
val_metrics = print_metrics(y_val, y_val_pred, "Validation")
test_metrics = print_metrics(y_test, y_test_pred, "Test")

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.show()

plot_confusion_matrix(y_val, y_val_pred, classes, "Validation")
plot_confusion_matrix(y_test, y_test_pred, classes, "Test")

# -----------------------------------------------------------------------------
# Performance Comparison with Different Hyperparameters

# Example: Modify max_depth and min_samples_split
dt_classifier = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
dt_classifier.fit(X_train_reshaped, y_train)
y_val_pred = dt_classifier.predict(X_val_reshaped)
y_test_pred = dt_classifier.predict(X_test_reshaped)

# Print Metrics for modified hyperparameters
val_metrics_mod = print_metrics(y_val, y_val_pred, "Validation with modified hyperparameters")
test_metrics_mod = print_metrics(y_test, y_test_pred, "Test with modified hyperparameters")

# Plot Confusion Matrix for modified hyperparameters
plot_confusion_matrix(y_val, y_val_pred, classes, "Validation with modified hyperparameters")
plot_confusion_matrix(y_test, y_test_pred, classes, "Test with modified hyperparameters")



