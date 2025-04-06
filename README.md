# Brain Tumor Detection with ResNet-18

This project implements a deep learning system for classifying brain tumors from MRI scans using a lightweight ResNet-18 model. It focuses on achieving high diagnostic performance with minimal computational overhead, making it viable for real-world clinical applications.
## Key Features

  - Binary classification: Tumor vs. No Tumor

  - Backbone: ResNet-18 for lightweight, high-speed inference

  - Enhanced training: Data augmentation, mixed precision, cosine LR scheduler

  - Evaluation metrics: Accuracy, F1-score, ROC-AUC, precision-recall curves

  - Visual insights: Confusion matrix, training/validation curves

  - Fast inference time: ~37 ms per image

## Overview

The repository is organized into modules for model training, evaluation, and visualization. Results are saved with classification metrics and plots for easy interpretation.
### Results Summary

  - Validation Accuracy: 94.12%

  - Test Accuracy: 90%

  - ROC-AUC Score: 0.94

  - Precision-Recall Average: 0.96

  - Inference Time: 36.89 ms per image

## Future Directions

   - Expand to multi-class tumor classification

   - Apply explainability tools like Grad-CAM

   - Explore ensemble methods for performance boost

   - Investigate deployment as a clinical web application
