
# Decision Tree Semi Supervised

import os
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib
matplotlib.use('Qt5Agg')  # Specify the backend
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to your dataset
data_dir = 'D:/Concordia/Applied AI/Summer 2024/Project - Place 365/COMP6721'
image_size = (256, 256)

# Parameter to specify number of images to load
num_images_to_load = 2500  # Change this number as needed

# Transformation for images (colorful images handled here)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()  # Converts images to tensors and retains 3 color channels (RGB)
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Ensure we do not load more images than available
num_images_to_load = min(num_images_to_load, len(dataset))
dataset, _ = random_split(dataset, [num_images_to_load, len(dataset) - num_images_to_load])

classes = dataset.dataset.classes
num_classes = len(classes)

# Function to split dataset
def split_dataset(dataset, test_size=0.2, validation_size=0.1, random_seed=42):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    test_split = int(np.floor(test_size * num_samples))
    validation_split = int(np.floor(validation_size * num_samples))
    
    test_indices = indices[:test_split]
    validation_indices = indices[test_split:test_split + validation_split]
    train_indices = indices[test_split + validation_split:]
    
    return train_indices, validation_indices, test_indices

# Function to convert DataLoader to numpy arrays
def loader_to_numpy(loader):
    data = []
    labels = []
    for images, targets in loader:
        data.append(images.view(images.size(0), -1).cpu().numpy())  # Flatten while retaining color info
        labels.append(targets.numpy())
    return np.vstack(data), np.hstack(labels)

# Split the dataset
train_indices, validation_indices, test_indices = split_dataset(dataset)

# DataLoaders
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=64, shuffle=True)
validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=64, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=64, shuffle=False)

# Convert DataLoaders to numpy arrays
train_data, train_labels = loader_to_numpy(train_loader)
validation_data, validation_labels = loader_to_numpy(validation_loader)
test_data, test_labels = loader_to_numpy(test_loader)

# Semi-supervised learning function
def semi_supervised_learning(train_data, train_labels, unlabeled_data, iterations=10, confidence_threshold=0.9):
    accuracies = []
    for i in range(iterations):
        if len(unlabeled_data) == 0:
            print("Unlabeled Data = 0")
            break
        print(f'Iteration = {i}')

        # Train the model on the labeled data
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(train_data, train_labels)
        
        # Calculate accuracy on the labeled data
        labeled_predictions = model.predict(train_data)
        labeled_accuracy = accuracy_score(train_labels, labeled_predictions)
        accuracies.append(labeled_accuracy)
        print(f'Iteration {i+1} - Labeled Data Accuracy: {labeled_accuracy:.4f}')
        
        # Predict pseudo-labels for the unlabeled data
        pseudo_labels = model.predict(unlabeled_data)
        probabilities = model.predict_proba(unlabeled_data)
        
        # Select high-confidence pseudo-labeled data
        high_confidence_mask = (np.max(probabilities, axis=1) >= confidence_threshold)
        high_confidence_data = unlabeled_data[high_confidence_mask]
        high_confidence_labels = pseudo_labels[high_confidence_mask]
        
        if len(high_confidence_data) == 0:
            print(f"Reached to High Confidence Accuracy Threshold >= {confidence_threshold}")
            break
        
        # Add high-confidence pseudo-labeled data to the training set
        train_data = np.vstack((train_data, high_confidence_data))
        train_labels = np.hstack((train_labels, high_confidence_labels))
        
        # Remove high-confidence pseudo-labeled data from unlabeled set
        unlabeled_data = unlabeled_data[~high_confidence_mask]
        
    return model, accuracies

# Specify the percentage of labeled and unlabeled data
labeled_percentage = 0.20
num_labeled = int(labeled_percentage * len(train_data))
num_unlabeled = len(train_data) - num_labeled

# Create labeled and unlabeled datasets
labeled_data, unlabeled_data = train_data[:num_labeled], train_data[num_labeled:]
labeled_labels = train_labels[:num_labeled]

# Perform semi-supervised learning
model, labeled_accuracies = semi_supervised_learning(labeled_data, labeled_labels, unlabeled_data)

# Evaluate the model on the test set
test_predictions = model.predict(test_data)
validation_predictions = model.predict(validation_data)

# Calculate metrics for test set
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')
test_conf_matrix = confusion_matrix(test_labels, test_predictions)

# Calculate metrics for validation set
validation_accuracy = accuracy_score(validation_labels, validation_predictions)
validation_precision = precision_score(validation_labels, validation_predictions, average='weighted')
validation_recall = recall_score(validation_labels, validation_predictions, average='weighted')
validation_f1 = f1_score(validation_labels, validation_predictions, average='weighted')
validation_conf_matrix = confusion_matrix(validation_labels, validation_predictions)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print('Test Confusion Matrix:')
print(test_conf_matrix)

print(f'Validation Accuracy: {validation_accuracy:.4f}')
print(f'Validation Precision: {validation_precision:.4f}')
print(f'Validation Recall: {validation_recall:.4f}')
print(f'Validation F1 Score: {validation_f1:.4f}')
print('Validation Confusion Matrix:')
print(validation_conf_matrix)

# Plot confusion matrix for test set
plt.figure(figsize=(10, 7))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot confusion matrix for validation set
plt.figure(figsize=(10, 7))
sns.heatmap(validation_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot labeled data accuracies
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(labeled_accuracies) + 1), labeled_accuracies, marker='o', linestyle='-')
plt.title('Labeled Data Accuracy During Semi-Supervised Learning')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
