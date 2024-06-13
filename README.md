# AppliedAI

This repository contains the code and methodologies for our course project on venue classification using image data. We implemented two different approaches: Decision Tree (supervised), Decision Tree (semi-supervised), and plan to implement Convolutional Neural Networks (CNN). This README provides an overview of the project, the structure of the repository, and instructions on how to run the code.

## Project Overview

In this project, we classify images into five distinct venue categories: airport terminal, movie theater, market, museum, and restaurant using supervied and semi-supervised decisiontree. we examine differnt values for three hyperparameters including max_depth, min_sample_split and min_sample_leaf.The dataset used is a subset of the Places365Standard dataset, consisting of 2,500 images divided into training, validation, and test sets. 

### Methodologies

1. **Decision Tree (Supervised)**
2. **Decision Tree (Semi-Supervised)**
3. **Convolutional Neural Network (CNN)**

The Decision Tree methodologies are implemented and included in this repository, while the CNN implementation will be added in the next weeks.

## Repository Structure

- **Code/DecisionTree**
  - `Main.py`: Main script for the Decision Tree classifier(supervised and sem-supervised).
  - This script defines and utilizes a custom decision tree classifier for image classification tasks. 
  The classifier is implemented in the `MyDecisionTree` class, which includes methods for loading images, 
  training the model, evaluating performance, and visualizing the decision tree. The script also performs 
  hyperparameter optimization using `GridSearchCV` to find the best parameters for the decision tree.

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
   - `Main_optimization.py`: this script defines and utilizes a custom decision tree classifier for image classification tasks. 
  The classifier is implemented in the `MyDecisionTree` class, which includes methods for loading images, 
  training the model, evaluating performance, and visualizing the decision tree. The script also performs 
  hyperparameter optimization using `GridSearchCV` to find the best parameters for the decision tree.
  - `Main_withoutNormalization.py`:  in this file the feature exteracted from images do not normalized.
  - `Main_withoutNormalization_optimization.py`: in this file the feature exteracted from images do not normalized.
     features value
 -`Semi_supervised.py`: semisupervised classifier with Decision tree model useing iteration

 

## Getting Started

### Prerequisites

- Python 3.x
- Scikit-learn
- Numpy
- Pandas

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
cd AppliedAI/Code/DecisionTree
```

### Run the desired script:

```bash
1- Download the dataset file from root
2- repalce the base path  in scripts.
3- run Main.py script in google colab  
```
Replace `Main.py` with any of the other scripts (`Main_withoutNormalization.py`, `Main_withoutNormalization_optimization.py`, `Main_optimization.py`) depending on your needs.

to run semi_suprvised.py:
1- Download the dataset2 file from root
2- repalce the data_dir path  in scripts.
3- run Semi_supervised.py script in google colab  

## Project Details

The detailed methodologies, dataset preparation, and results are documented in the project report. Below is a brief overview:

- **Database**: The dataset is a part of the publicly available Places365Standard dataset, with images resized to 256x256 pixels and normalized.
- the original dataset  is a large-scale venue classification dataset comprising approximately 1.8 million training images and 36,500 validation images across 365 classes. The original images are colorful and 256x256 pixelsin size and are available for download from the official Places
website( Places365 website: http://places2.csail.mit.edu/download.html)
- **Model**: Decision Tree Classifier from the Scikit-learn library with hyperparameter tuning using GridSearchCV.
- **Results**: Performance metrics including accuracy, precision, recall, and F1 score and confusion matrix are evaluated for both training and validation sets.

## Authors and Contacts

- Matin Mansouri: [matin.mansouri23@gmail.com](mailto:matin.mansouri23@gmail.com)
- Zahed Ebrahimi: [zahedebrahimi89@gmail.com](mailto:zahedebrahimi89@gmail.com)
- Samane Vazrrian: [samane.vazirian@gmail.com](mailto:samane.vazirian@gmail.com)

