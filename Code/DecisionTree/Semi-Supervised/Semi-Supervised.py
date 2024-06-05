import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_images(base_path, classes, image_size):
    # Initialize lists to store images and labels
    images = []
    labels = []
    for class_folder in classes:
        folder_path = os.path.join(base_path, class_folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path).convert('RGB')
            image = image.resize(image_size)
            image_array = np.array(image)
            images.append(image)
            labels.append(class_folder)  # Use the folder name as the label

    # Convert images to a numpy array and normalize
    images = np.array(images, dtype=np.float32) / 255.0
    images_flattened = images.reshape(len(images), -1)

    # Encode class labels to numerical values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images_flattened, labels

# Define paths and classes
base_path = "/content/drive/MyDrive/ColabNotebooks/DataSet/"
classes = ["airport_terminal", "market", "movie_theater", "museum", "restaurant"]
image_size = (256, 256)

# Load train images and labels
X_train, Y_train = load_images(os.path.join(base_path, "train"), classes, image_size)

# Load validation images and labels
X_val, Y_val = load_images(os.path.join(base_path, "val"), classes, image_size)

def semi_supervised_self_training(X, y, labeled_portion=0.2,
                                  threshold=0.9, criterion='threshold'
                                 ):

    # Separate labeled and unlabeled data
    num_unlabeled = int(len(y) * (1 - labeled_portion))
    unlabeled_indices = np.random.choice(np.arange(len(y)), num_unlabeled, replace=False)
    Y_semi = np.copy(y)
    Y_semi[unlabeled_indices] = -1
    
    # Create the SelfTrainingClassifier with a DecisionTreeClassifier base classifier
    base_dtc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=10)
    self_training_clf = SelfTrainingClassifier(base_dtc, criterion=criterion, threshold=threshold)

    # Fit the self-training classifier on the semi-labeled data
    self_training_clf.fit(X, Y_semi)

    return self_training_clf

stc_model = semi_supervised_self_training(X_train, Y_train)

# Evaluate the model on the validation data
val_pred = stc_model.predict(X_val)
accuracy = accuracy_score(Y_val, val_pred)
precision = precision_score(Y_val, val_pred, average='macro')
recall = recall_score(Y_val, val_pred, average='macro')
f1 = f1_score(Y_val, val_pred, average='macro')
conf_matrix = confusion_matrix(Y_val, val_pred, sample_weight=None)
class_report = classification_report(Y_val, val_pred)

# Print the metrics
print("val Accuracy:", accuracy)
print("val Precision:", precision)
print("val Recall:", recall)
print("val F1 Score:", f1)
print("val Confusion Matrix:\n", conf_matrix)
print("val Classification Report:\n", class_report)
