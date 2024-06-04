
# Decision Tree supervised

import os
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load Images Function to load iamges from Drive and put the in array
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
        #print(image_array.shape)
        images.append(image)
        labels.append(class_folder)  # Use the folder name as the label


 # Convert images to a numpy array and normalize(# Normalize pixel values to [0, 1])
 images = np.array(images, dtype=np.float32) / 255.0
 #print(images.shape)

 # Flatten the images (samples x features) to conver array to 1D array
 images_flattened = images.reshape(len(images), -1)


 # Encode class lables to numerical values
 label_encoder = LabelEncoder()
 labels = label_encoder.fit_transform(labels)

 # return an array of features(image pixele)
 # each row of this array present exteracted featurs from one train iamges
 # and labels array pressent and array of class lables
 return images_flattened, labels

# Define paths and classes
base_path = "/content/drive/MyDrive/ColabNotebooks/DataSet/"  # Update this with the correct path
classes = ["airport_terminal", "market", "movie_theater", "museum", "restaurant"]
image_size = (256, 256)
X_train = []
Y_train = []


# Load  train images and labels
X_train,Y_train =load_images(os.path.join(base_path,"train"),classes,image_size)

# Load  Val images and labels
X_val = []
Y_val = []
X_val,Y_val =load_images(os.path.join(base_path, "val"),classes,image_size)

# class lable Encoding
le = LabelEncoder()
lable_classes = le.fit_transform(classes)


# Train Decision Tree Classifier
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, Y_train)
#tree.plot_tree(dtc)

# Define the parameter grid
param_grid = {
    'min_samples_split': [2, 5, 10]
}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation data
validation_accuracy = best_model.score(X_val, Y_val)
print("Validation Accuracy with best model:", validation_accuracy)









