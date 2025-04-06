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
## Results

### Training vs Validation Curves
![image](https://github.com/user-attachments/assets/2a8ed615-46e4-4070-8277-394e7a03f479)

The training and validation accuracy/loss curves demonstrate consistent learning without signs of overfitting. The model achieves convergence within 30 epochs, showing stable generalization.

### Confusion Matrix

![Confusion_matrix](https://github.com/user-attachments/assets/88164a2c-e36c-4535-afc0-e3e8ffe0cda0)

The confusion matrix illustrates the distribution of correct and incorrect predictions for each class. The model shows high sensitivity and specificity, with minimal misclassifications between tumor and non-tumor classes.

### ROC Curve

![ROC](https://github.com/user-attachments/assets/89ae5ca7-dac4-404b-aff0-c31d1e365a30)

The ROC curve shows the trade-off between true positive rate and false positive rate. An AUC close to 1.0 indicates excellent classification performance, even under threshold variations.

### Precision-Recall Curve

![Precision_recall](https://github.com/user-attachments/assets/b487eaab-a008-484a-926e-3d525aa0e0aa)

This curve highlights the model’s ability to maintain high precision and recall, especially important for imbalanced datasets where false negatives are costly in medical diagnostics.

### Average Inference Time

  36.89 ms/image
  
  The model achieves fast inference, making it viable for real-time or clinical deployment on GPU-based systems.
## Future Directions

   - Expand to multi-class tumor classification

   - Apply explainability tools like Grad-CAM

   - Explore ensemble methods for performance boost

   - Investigate deployment as a clinical web application
