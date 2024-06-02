
# Decision Tree supervised

import os
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


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
 # each row of this array present exteracted features from one train iamges
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

length = len(X_train)
print (length)
# Train Decision Tree Classifier
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, Y_train)
#tree.plot_tree(dtc)

# Get the depth of the tree
tree_depth = dtc.get_depth()
print("Depth of the tree:", tree_depth)
# print a nicer tree using graphviz


# Make predictions on the train set
y_train_pred = dtc.predict(X_train)


# Make predictions on the val set
y_val_pred = dtc.predict(X_val)

# Evaluate the model  # Use 'macro' for multi-class

###Train
accuracy = accuracy_score(Y_train, y_train_pred)
precision = precision_score(Y_train, y_train_pred, average='macro') 
recall = recall_score(Y_train, y_train_pred, average='macro')        
f1 = f1_score(Y_train, y_train_pred, average='macro')                
conf_matrix = confusion_matrix(Y_train, y_train_pred)
class_report = classification_report(Y_train, y_train_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

###val
accuracy = accuracy_score(Y_val, y_val_pred)
precision = precision_score(Y_val, y_val_pred, average='macro') 
recall = recall_score(Y_val, y_val_pred, average='macro')        
f1 = f1_score(Y_val, y_val_pred, average='macro')                
conf_matrix = confusion_matrix(Y_val, y_val_pred)
class_report = classification_report(Y_val, y_val_pred)

# Print the metrics
print("val Accuracy:", accuracy)
print("val Precision:", precision)
print("val Recall:", recall)
print("val F1 Score:", f1)
print("val Confusion Matrix:\n", conf_matrix)
print("val Classification Report:\n", class_report)










