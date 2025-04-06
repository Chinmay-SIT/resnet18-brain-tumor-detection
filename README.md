# resnet18-brain-tumor-detection
🧠 Brain tumor detection using ResNet-18 on MRI scans with high accuracy and fast inference. Includes training code and evaluation metrics.

This project presents an accurate and efficient approach to brain tumor classification on MRI scans using a ResNet-18 model enhanced with data augmentation, cosine learning rate scheduling, and mixed precision training.

## Highlights
- Achieved **94.12% validation accuracy** and **90% test accuracy**
- Lightweight and fast: **36.89 ms inference time**
- Evaluated with ROC-AUC, PR curves, and confusion matrix
- LaTeX paper and training code included

## Project Structure
- `code/` - Python scripts for training and evaluation
- `figures/` - Plots and visualizations
- `models/` - Saved PyTorch model
- `paper/` - LaTeX source for the research paper

## Setup
```bash
git clone https://github.com/yourusername/resnet18-brain-tumor-detection.git
cd resnet18-brain-tumor-detection
pip install -r requirements.txt

