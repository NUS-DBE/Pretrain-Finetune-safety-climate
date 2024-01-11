import numpy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import  torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

# 评估模型函数
def evaluate_model(model, test_dataset, batch_size):
    # 将模型设置为评估模式
    model.eval()

    # DataLoader for our test set
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # List to store predictions and actual labels
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)

            # 使用sigmoid函数将输出转换为介于0和1之间的概率
            outputs=torch.argmax(outputs, dim=1)
            probs = outputs.cpu().detach().item()
            # Save the probabilities for AUC calculation
            predictions.append(probs)
            true_labels.append(labels.cpu().item() )

    # Convert lists of tensors to single tensors
    # predictions = torch.cat(predictions).numpy()
    # true_labels = torch.cat(true_labels).numpy()
    predictions =numpy.array(predictions)
    true_labels=numpy.array(true_labels).astype(int)

    binary_predictions=predictions
    # binary_predictions = (predictions > 0.5).astype(int)
    auc = roc_auc_score(true_labels, predictions)
    # Calculate metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    f1 = f1_score(true_labels, binary_predictions)
    conf_matrix = confusion_matrix(true_labels, binary_predictions)
    balance_accuracy=balanced_accuracy_score(true_labels, binary_predictions)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),  # Convert to list for easy printing or storing
        'balance_accuracy':balance_accuracy,
        'auc': auc
    }
