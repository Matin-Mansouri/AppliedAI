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
  - `Main_withoutNormalization.py`: Main script for the Decision Tree classifier(supervised and sem-supervised) without normalization.
  - `Main_withoutNormalization_optimization.py`:Script for examine hyperparameter values in order to optimize model without Normalization on 
     features value
  - `Main_optimization.py`: Script for examine hyperparameter values inorderto optimize model.

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
python Main.py
```
Replace `Main.py` with any of the other scripts (`Main_withoutNormalization.py`, `Main_withoutNormalization_optimization.py`, `Main_optimization.py`) depending on your needs.

## Project Details

The detailed methodologies, dataset preparation, and results are documented in the project report. Below is a brief overview:

- **Database**: The dataset is a part of the publicly available Places365Standard dataset, with images resized to 256x256 pixels and normalized.
- **Model**: Decision Tree Classifier from the Scikit-learn library with hyperparameter tuning using GridSearchCV.
- **Results**: Performance metrics including accuracy, precision, recall, and F1 score and confusion matrix are evaluated for both training and validation sets.

## Authors and Contacts

- Matin Mansouri: [matin.mansouri23@gmail.com](mailto:matin.mansouri23@gmail.com)
- Zahed Ebrahimi: [zahedebrahimi89@gmail.com](mailto:zahedebrahimi89@gmail.com)
- Samane Vazrrian: [samane.vazirian@gmail.com](mailto:samane.vazirian@gmail.com)

