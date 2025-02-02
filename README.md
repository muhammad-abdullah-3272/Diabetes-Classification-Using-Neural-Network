# Diabetes Classification Using Neural Networks

## Overview
This project implements a **Neural Network-based classification model** for predicting diabetes. The model is trained on structured health data and optimized using backpropagation and gradient descent. 

> **Note:** This is a **from-scratch implementation**, meaning no external deep learning frameworks like TensorFlow or PyTorch were used. The neural network, backpropagation, and gradient descent were all implemented manually in Python.

## Table of Contents
- [Dataset Details](#dataset-details)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Model Selection](#model-selection)
- [Gradient Checking](#gradient-checking)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Final Results](#final-results)
- [Conclusion](#conclusion)

## Dataset Details
The dataset consists of **769 examples**, divided as follows:

| Dataset Name       | Number of Examples |
|--------------------|-------------------|
| Training Set      | 539               |
| Validation Set    | 115               |
| Test Set         | 115               |

## Data Processing
### Pre-processing
- Checked for **NaN values** (none found).
- Converted object data types (dates/times) into numerical format.
- Applied **feature scaling** using Min-Max normalization.

### Feature Scaling Formula
$$x' = \frac{x - min(x)}{max(x) - min(x)}$$

## Model Architecture
- The Neural Network consists of:
  - **Input Layer:** 3 neurons (corresponding to dataset features)
  - **Hidden Layer:** 7 neurons with ReLU activation
  - **Output Layer:** 1 neuron with Sigmoid activation for binary classification
- Fully connected neural network with manual forward and backward propagation.

## Model Selection
- The **activation functions** tested:
  - **Sigmoid in all layers** → resulted in slow convergence
  - **ReLU in hidden layers, Sigmoid in output** → best performance
- **Regularization parameter (λ) = 0** was optimal after testing values from 0 to 1.
- **3-layered vs. 4-layered networks**: The **3-layered** network generalized better due to the limited dataset size.

## Gradient Checking
- Gradient checking was performed for the first **2 iterations** to ensure correct backpropagation implementation.
- The calculated gradients matched numerical gradients within a **10⁻⁵ to 10⁻⁷** error range, validating the correctness of the implementation.

## Model Training and Evaluation
### Training Details
- **3-layered Neural Network** with **20 neurons per hidden layer**.
- **Learning rate (α) = 4**
- **Training Iterations = 20**
- Cost minimized through manual gradient descent optimization.

### Model Performance
| Metric  | Training Loss | Validation Loss | Test Loss |
|---------|--------------|----------------|-----------|
| Cost Errors | 0.255 | 0.271 | 0.214 |
| Accuracy (%) | 89.7 | 90.4 | 93.0 |

## Final Results
| Model Type       | Training Accuracy (%) | Validation Accuracy (%) | Test Accuracy (%) |
|------------------|----------------------|----------------------|------------------|
| Neural Network  | 89.7 | 90.4 | 93.0 |

## Conclusion
- This project **implements a neural network from scratch**, demonstrating a deep understanding of forward and backward propagation.
- Neural Networks significantly improved diabetes classification compared to previous regression models.
- **93% test accuracy** was achieved with optimal hyperparameters.
- Future improvements include **expanding the dataset**, **tuning hyperparameters further**, and **testing deeper architectures** for enhanced performance.