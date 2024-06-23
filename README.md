# AppliedAI

This repository contains the code and methodologies for our course project on venue classification using image data. We implemented two different approaches: Decision Tree (supervised), Decision Tree (semi-supervised), and plan to implement Convolutional Neural Networks (CNN). This README provides an overview of the project, the structure of the repository, and instructions on how to run the code.

## Project Overview

In this project, we classify images into five distinct venue categories: airport terminal, movie theater, market, museum, and restaurant using supervised and semi-supervised decision-tree. we examine different values for three hyperparameters including max_depth, min_sample_split, and min_sample_leaf. The dataset used is a subset of the Places365Standard dataset, consisting of 2,500 images divided into training, validation, and test sets. 

### Methodologies

1. **Decision Tree (Supervised)**
2. **Decision Tree (Semi-Supervised)**
3. **Convolutional Neural Network (CNN)**


## Repository Structure

- **Code/DecisionTree**
  - `Main.py`: Main script for the Decision Tree classifier (supervised and semi-supervised).
    - This script defines and utilizes a custom decision tree classifier for image classification tasks. The classifier is implemented in the `MyDecisionTree` class, which includes methods for loading images, training the model, evaluating performance, and visualizing the decision tree. The script also performs hyperparameter optimization using `GridSearchCV` to find the best parameters for the decision tree.

  Classes and Methods:
  1. `MyDecisionTree`: A custom decision tree classifier class with the following methods:
      - `load_images`: Loads images from specified directories, resizes them, and converts them to numpy arrays.
      - `plot_tree`: Plots and saves a visual representation of the decision tree using graphviz.
      - `decisiontree_evaluate`: Evaluates and prints various performance metrics for the classifier.
      - `decisiontree_optimization`: Optimizes decision tree hyperparameters using `GridSearchCV` and logs performance metrics.
      - `plot_performance`: Plots performance improvement per hyperparameter combination.
      - `semi_supervised_learning`: Perform semi-supervised learning using a decision tree classifier.

  Main Script Workflow:
  1. Define the base path to the dataset and the class names.
  2. Create an instance of the `MyDecisionTree` class.
  3. Load training and validation images and labels.
  4. Perform hyperparameter optimization for `min_samples_split`, `min_samples_leaf`, and `max_depth`.
  5. Evaluate the optimized models on the validation set and plot performance metrics.

  - `Main_optimization.py`: This script defines and utilizes a custom decision tree classifier for image classification tasks. The classifier is implemented in the `MyDecisionTree` class, which includes methods for loading images, training the model, evaluating performance, and visualizing the decision tree. The script also performs hyperparameter optimization using `GridSearchCV` to find the best parameters for the decision tree.
  - `Main_withoutNormalization.py`: In this file, the features extracted from images are not normalized.
  - `Main_withoutNormalization_optimization.py`: In this file, the features extracted from images are not normalized.
  - `Semi_supervised.py`: Defines a semi-supervised classifier with a Decision Tree model using iteration.

- **Code/CNN**
  - `CNN.py`: Main script for the Convolutional Neural Network classifier with the best hyperparameters.
    - This script sets up the architecture for the CNN, trains the model, and evaluates its performance on the dataset.

  Classes and Methods:
  1. `ImprovedCNN`: A custom CNN classifier class with the following methods:
      - `load_data`: Loads images from specified directories, resizes them and converts them to tensors.
      - `build_model`: Defines the architecture of the CNN using PyTorch.
      - `train_model`: Trains the CNN on the training dataset.
      - `evaluate_model`: Evaluates the trained CNN on the validation dataset.
      - `save_model`: Saves the trained model to a file.
      - `load_model`: Loads a pre-trained model from a file.

  Main Script Workflow:
  1. Define the base path to the dataset and the class names.
  2. Create an instance of the `ImprovedCNN` class.
  3. Load training and validation images and labels.
  4. Build the CNN model.
  5. Train the model on the training dataset.
  6. Evaluate the model on the validation dataset.
  7. Save the trained model.

  Hyperparameter Tuning Folders:
  - **Activation function**: Contains scripts and results for varying activation functions.
  - **Batchsize**: Contains scripts and results for varying batch sizes.
  - **Dropout**: Contains scripts and results for varying dropout rates.
  - **FC**: Contains scripts and results for varying the number of fully connected layers.
  - **Learning rate**: Contains scripts and results for varying learning rates.
  - **Optimizer**: Contains scripts and results for varying optimizers.
  - **Without batch normalization**: Contains scripts and results for experiments without batch normalization.

  Each folder includes:
  - Python script: Named to indicate the specific hyperparameter and its value.
  - Result text file: Contains the performance metrics and results of the corresponding script.
  - `results.txt`: A document summarizing the results of the CNN experiments.



 

## Getting Started

### Prerequisites

- Python 3.x
- Scikit-learn
- Numpy
- Pandas
- torch
- torch.nn 
- torch.optim 
- torchvision
- torchvision.transforms 
- os
- torch.utils.data.DataLoader
- torch.optim.lr_scheduler.StepLR
- sklearn.metrics (precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)
- matplotlib.pyplot 
- seaborn 


### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AppliedAI.git
cd AppliedAI
```

### Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Code

### Navigate to the appropriate directory:

```bash
cd AppliedAI/Code
```


 
### Project Details :

The detailed methodologies, dataset preparation, and results are documented in the project report. Below is a brief overview:

- **Database**: The dataset is a part of the publicly available Places365Standard dataset, with images 256x256 pixels and normalized.The images are 
  organized into folders according
  to their respective classes and are in JPEG format. we selected five specific classes from the original dataset: "airport_terminal" "market" 
  "movie_theater" "museum" and "restaurant" and for each class, 500 images were randomly chosen from the original dataset.
  the original dataset  is a large-scale venue classification dataset comprising approximately 1.8 million training images and 36,500 validation 
  images across 365 classes. The original images are colorful and 256x256 pixelsin size and are available for download from the official Places
  website( Places365 website: http://places2.csail.mit.edu/download.html)

- Model:
  1. The Decision Tree Classifier from the Scikit-learn library with hyperparameter tuning using GridSearchCV. The best-evaluated hyperparameters include max_depth=10, min_samples_split=20, and min_samples_leaf=13. The values evaluated among these are max_depth: [4, 7, 8, 10, 12, 14], min_samples_split: [4, 8, 11, 13], and min_samples_leaf: [15, 20, 30].
  2. The semi-supervised Decision Tree Classifier.
  3. The Convolutional Neural Network (CNN) implemented in PyTorch. The best results are obtained with the following hyperparameters:
      - Activation function: ReLU
      - Batchsize: 64
      - Dropout: 0.2
      - Fully Connected Layers (FC): 5 layers
      - Learning rate: 0.0001
      - Optimizer: ADAM
      - With batch normalization

            

- Result: Performance metrics including accuracy, precision, recall, F1 score, and confusion matrix are evaluated for training and validation sets. The accuracy for test data for the Decision Tree classifier and semi-supervised Decision Tree classifier is 0.31 and 0.33, respectively.

For the CNN results:
- Test Accuracy: 0.68
- Test Precision: 0.6819
- Test Recall: 0.6820
- Test F1 Score: 0.6809
- Train Loss: 0.7212
- Val Loss: 0.7909
- Train Accuracy: 0.74
- Val Accuracy: 0.71
- Train Precision: 0.7369
- Val Precision: 0.7175
- Train Recall: 0.7419
- Val Recall: 0.7100
- Train F1 Score: 0.7374
- Val F1 Score: 0.7107

`
## Authors and Contacts


- Matin Mansouri: [matin.mansouri23@gmail.com](mailto:matin.mansouri23@gmail.com)
- Zahed Ebrahimi: [zahedebrahimi89@gmail.com](mailto:zahedebrahimi89@gmail.com)
- Samane Vazrrian: [samane.vazirian@gmail.com](mailto:samane.vazirian@gmail.com)
'
