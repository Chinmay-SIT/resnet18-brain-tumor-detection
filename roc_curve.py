from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Assuming binary classification: 0 = No Tumor, 1 = Tumor
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 'Tumor'
        all_probs.extend(probs.cpu().numpy())

# Binarize labels
y_true = [1 if label == 1 else 0 for label in all_labels]

# ROC & AUC
fpr, tpr, _ = roc_curve(y_true, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid()
plt.show()
