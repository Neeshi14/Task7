# Task 7
## 1)
This project analyzes a thyroid dataset to predict thyroid disease status using deep learning techniques.

## Dataset

* **Name:**  Thyroid Cancer Risk Dataset
* **Dataset name:** Thyroid_cancer_risk_data.csv
* **Description:**
* There are 17 features

Here’s a brief one-line explanation for each feature of dataset:
Patient_ID (int): Unique identifier for each patient.

Age (int): Age of the patient.

Gender (object): Patient’s gender (Male/Female).

Country (object): Country of residence.

Ethnicity (object): Patient’s ethnic background.

Family_History (object): Whether the patient has a family history of thyroid cancer (Yes/No).

Radiation_Exposure (object): History of radiation exposure (Yes/No).

Iodine_Deficiency (object): Presence of iodine deficiency (Yes/No).

Smoking (object): Whether the patient smokes (Yes/No).

Obesity (object): Whether the patient is obese (Yes/No).

Diabetes (object): Whether the patient has diabetes (Yes/No).

TSH_Level (float): Thyroid-Stimulating Hormone level (µIU/mL).

T3_Level (float): Triiodothyronine level (ng/dL).

T4_Level (float): Thyroxine level (µg/dL).

Nodule_Size (float): Size of thyroid nodules (cm).

Thyroid_Cancer_Risk (object): Estimated risk of thyroid cancer (Low/Medium/High).

Diagnosis (object): Final diagnosis (Benign/Malignant).


## Project Overview

* **Goal:**  classify nodules as benign/malignant
* **Methods:** The neural networks are used.
* **Results:** The model achieved 82.87% accuracy on the test set.

In these the simple linear neural network using the pytorch.

# Pytorch basic knowledge
  PyTorch is a free, open-source machine learning framework that helps developers build and train deep learning models. It's written in Python and C++, and is known for its 
  ease of use and flexibility. 



## 2)

# MNIST Handwritten Digit Classification using PyTorch

This project implements a convolutional neural network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.  The model is trained to recognize digits from 0 to 9.

## Dataset

* **Name:** MNIST Handwritten Digits
* **Source:** The MNIST dataset is a widely used benchmark dataset in computer vision. It is available through the `torchvision.datasets` module in PyTorch.
* **Description:** The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. The images are grayscale and 28x28 pixels in size.

## Project Overview

* **Goal:** The goal of this project is to train a CNN model that can accurately classify handwritten digits from the MNIST dataset.
* **Methods:** A convolutional neural network (CNN) architecture is used. The network consists of convolutional layers, pooling layers, and fully connected layers. ReLU activation functions are used after the convolutional layers.  A softmax function is used for the final output to obtain probabilities for each digit class. The model is trained using the Adam optimizer and the CrossEntropyLoss loss function.
* **Results:** The trained model achieves 97% accuracy on the MNIST test set.  

