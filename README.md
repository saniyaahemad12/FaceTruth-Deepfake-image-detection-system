# FaceTruth-Deepfake Detection for Images

## Introduction-

Deepfakes are synthetic media generated using deep learning techniques, often to create hyper-realistic fake images and videos.
They can pose significant threats to security, trust, and authenticity in digital communications.
This project focuses on detecting deepfake images using machine learning and deep learning models, aiming to distinguish between real and fake images based on subtle inconsistencies introduced during synthetic generation.

## Problem Statement
The primary challenge is to build a model that can analyze an image and classify it as real or fake (deepfake).
Deepfakes often contain minute artifacts such as unnatural skin textures, inconsistent lighting, irregular facial features, and other anomalies not easily detected by the human eye.
The goal is to automate this detection process reliably and accurately.

## Methodology
### 1.Data Collection and Preprocessing

A dataset containing both real and fake images is collected (e.g., FaceForensics++, DeepFake Detection Dataset).

Images are preprocessed through resizing, normalization, and augmentation techniques to enhance the robustness of the model.

### 2.Model Architecture

A Convolutional Neural Network (CNN) architecture is employed for feature extraction and classification.

Transfer learning techniques (e.g., using pretrained models like EfficientNet, XceptionNet) can be used to improve performance.

Regularization methods such as dropout and data augmentation are applied to prevent overfitting.

### 3.Training

The model is trained on labeled data using appropriate loss functions (e.g., Binary Crossentropy) and optimization algorithms (e.g., Adam Optimizer).

Performance is monitored using metrics like accuracy, precision, recall, and F1 score.

### 4.Prediction

Given a new input image, the model predicts whether it is a real or deepfake image based on learned patterns.
![WhatsApp Image 2025-04-20 at 11 39 08 PM](https://github.com/user-attachments/assets/c71a73fa-5bf5-44c2-accd-a6e90464d64e)

![WhatsApp Image 2025-04-20 at 11 47 06 PM](https://github.com/user-attachments/assets/36fd62b8-c61d-4a54-aa11-2e42c09256d6)


## Key Techniques Used
CNN-based Feature Extraction

Transfer Learning with pretrained models

Image Data Augmentation

Regularization to Prevent Overfitting

Binary Classification Metrics for Evaluation

Flask-based Web Deployment
