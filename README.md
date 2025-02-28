# Task 7
# Basic Konwledge about the pytorch
1. What is PyTorch?

PyTorch is a popular open-source machine learning framework, especially well-suited for deep learning. You could mention that it's developed by Facebook's AI Research lab and is known for its flexibility and ease of use.

2. Key Concepts to Explain :

->Tensors: Explain that tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with added capabilities for automatic differentiation (which is essential for training neural networks). 

->Modules (nn.Module): If you're using custom neural network architectures, explain that nn.Module is the base class for building neural networks in PyTorch. It helps organize the network's layers and parameters.  You can say something like, "The neural network model is defined as a class inheriting from nn.Module, which allows for modular design and management of the network's components."

** ->Layers: Briefly describe the types of layers you used in your network (e.g., convolutional layers, linear layers, recurrent layers).  For example:

1.Convolutional Layers (Conv2d, Conv1d): "Convolutional layers are used to extract features from images (2D) or sequences (1D)."
Linear Layers (Linear): "Linear layers (also called fully connected layers) connect all neurons in one layer to all neurons in the next layer. They are often used at the end of a network for classification."

2.Pooling Layers (MaxPool2d, AvgPool2d): "Pooling layers reduce the spatial dimensions of feature maps (e.g., images), making the network more efficient and robust to small variations in the input."

3.Dropout: "Dropout is a regularization technique that helps prevent overfitting by randomly deactivating neurons during training."

4.Loss Function: Explain that the loss function measures the difference between the model's predictions and the actual target values.  Mention the specific loss function you used (e.g., Cross-Entropy Loss for classification, Mean Squared Error for regression).  "The Cross-Entropy Loss function is used to measure the difference between the predicted probabilities and the true class labels."

5.Optimizer: Explain that the optimizer is responsible for updating the model's parameters (weights) to minimize the loss function.  Mention the optimizer you used (e.g., Adam, SGD).  "The Adam optimizer is used to adjust the model's parameters during training, aiming to minimize the loss."

->Training Loop: Briefly describe the training process:

-Forward pass: The input data is fed to the model to get predictions.

-Loss calculation: The loss function compares predictions to the true values.

-Backpropagation: The gradients of the loss with respect to the model's parameters are calculated.

-Optimization: The optimizer updates the parameters based on the gradients.

-Evaluation: I evaluated the model's performance by accuracy.  "The model's performance is evaluated using accuracy on the test dataset, which measures the percentage of correctly classified samples."
  


## 1)
#  This project analyzes a thyroid dataset to predict thyroid disease status using deep learning techniques.

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


1.  **Run the model:**

    Execute the Python script to train and evaluate the model. The script will perform the following steps:

    * Load and preprocess the dataset.
    * Convert categorical features to numerical.
    * Split the dataset into training and testing sets.
    * Define and train the neural network model.
    * Evaluate the model's performance.
    * Display the accuracy


## Model Architecture

* **Type:** Feedforward Neural Network
* **Input Layer:** 14 features
* **Hidden Layers:** Two hidden layers with 64 and 32 neurons, respectively, using ReLU activation.
* **Output Layer:** 2 neurons (binary classification: benign or malignant)
* **Activation Function:** ReLU for hidden layers, Softmax (implicitly within CrossEntropyLoss) for the output layer.
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (learning rate 0.001)

## Training Details

* **Epochs:** 3
* **Batch Size:** 32
* **Learning Rate:** 0.001
* **Train/Test Split:** 80/20

## Evaluation

The model's performance is evaluated on the test set, and the following metrics are reported:

* **Training Accuracy:** 82%
* **Testing Accuracy:**  82%
* **Classification Report:** Precision, recall, and F1-score for each class.

## Results

The model achieves an accuracy of approximately 82.87% in classifying thyroid cancer risk.

In these the simple linear neural network using the pytorch.





## 2)

# MNIST Handwritten Digit Classification using PyTorch

This project implements a convolutional neural network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.  The model is trained to recognize digits from 0 to 9.

## Dataset

* **Name:** MNIST Handwritten Digits
* **Source:** The MNIST dataset is a widely used benchmark dataset in computer vision. It is available through the `torchvision.datasets` module in PyTorch.
* **Description:** The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. The images are grayscale and 28x28 pixels in size.

1.  **Run the model:**

    Execute the Python script to train and evaluate the model. The script will perform the following steps:

    * Download and load the MNIST dataset.
    * Define the CNN model architecture.
    * Train the model.
    * Evaluate the model's performance on the test set.
    * Visualize predictions on sample images.


## Model Architecture

* **Type:** Convolutional Neural Network (CNN)
* **Convolutional Layers:**
    * `Conv2d(1, 32, 3, 1)`
    * `Conv2d(32, 64, 3, 1)`
    * `Conv2d(64, 128, 3, 1)`
    * `Conv2d(128, 256, 3, 1)`
    * `Conv2d(256, 750, 3, 1)`
* **Activation Function:** ReLU
* **Pooling Layers:** Max pooling (kernel size 2)
* **Dropout Layers:** Dropout2d (0.25, 0.5, 0.25)
* **Fully Connected Layers:**
    * `Linear(750, 100)`
    * `Linear(100, 10)`
* **Output Activation:** Softmax
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (learning rate 0.001)

## Training Details

* **Epochs:** 3
* **Batch Size (Train):** 164
* **Batch Size (Test):** 100
* **Learning Rate:** 0.001
* **Normalization:** `transforms.Normalize((0.1307,), (0.3081,))`

## Evaluation

The model's performance is evaluated on the test set, and the following metrics are reported:
* **Training Accuracy:** 96%
* **Testing Accuracy:** 97%

## Results

The CNN model achieves an accuracy of 97.32% on the MNIST test set, demonstrating its effectiveness in classifying handwritten digits.

## Visualizations

The script includes visualizations to display sample images from the test set along with their predicted and true labels.


