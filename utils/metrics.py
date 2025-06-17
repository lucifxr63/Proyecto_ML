from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import torch


def compute_metrics(y_true, y_pred):
    """Compute F1-score, AUC-ROC and confusion matrix."""
    y_true = y_true.cpu().numpy()
    y_pred_prob = torch.softmax(y_pred, dim=1).detach().cpu().numpy()
    y_pred_labels = y_pred_prob.argmax(axis=1)
    f1 = f1_score(y_true, y_pred_labels)
    try:
        auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred_labels)
    return f1, auc, cm
