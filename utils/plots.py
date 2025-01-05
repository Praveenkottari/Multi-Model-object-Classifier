import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_curve

def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes, rotation =45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize = 16)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Increase padding on the bottom to avoid x-labels being cut off
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path)
    plt.close()


def plot_sample_predictions(inputs, labels, preds, classes, save_path):
    # Plot a few sample images with predictions
    plt.figure(figsize=(12,6))
    for i in range(min(6, len(inputs))):
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(f"True: {classes[labels[i]]}\nPred: {classes[preds[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_f1_confidence(y_true, y_probs, classes, save_path):
    """
    Compute F1 at different confidence thresholds and plot.
    y_probs: shape [N, num_classes], softmax probabilities
    """
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0, 1, 50)
    f1_all = []
    f1_classes = {cls: [] for cls in classes}
    y_true_arr = np.array(y_true)

    for t in thresholds:
        y_pred = np.argmax(y_probs, axis=1)
        max_probs = np.max(y_probs, axis=1)
        y_pred[max_probs < t] = -1  # Treat as no prediction
        valid_idx = (y_pred != -1)

        if np.sum(valid_idx) == 0:
            f1_all.append(0.0)
            for c in classes:
                f1_classes[c].append(0.0)
            continue

        # Overall macro F1
        f1_all.append(f1_score(y_true_arr[valid_idx], y_pred[valid_idx], average='macro', zero_division=0))

        # Per-class F1 (one-vs-rest)
        for i, c in enumerate(classes):
            y_true_class = (y_true_arr[valid_idx] == i).astype(int)
            y_pred_class = (y_pred[valid_idx] == i).astype(int)
            f1_c = f1_score(y_true_class, y_pred_class, average='binary', zero_division=0)
            f1_classes[c].append(f1_c)

    plt.figure()
    plt.plot(thresholds, f1_all, linewidth=4,
             label=f'all classes {np.max(f1_all):.2f} at {thresholds[np.argmax(f1_all)]:.3f}')
    for c in classes:
        plt.plot(thresholds, f1_classes[c], label=c)
    plt.xlabel('Confidence')
    plt.ylabel('F1')
    plt.title('F1-Confidence Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_precision_confidence(y_true, y_probs, classes, save_path):
    from sklearn.metrics import precision_score
    thresholds = np.linspace(0, 1, 50)
    precision_all = []
    precision_classes = {cls: [] for cls in classes}
    y_true_arr = np.array(y_true)

    for t in thresholds:
        y_pred = np.argmax(y_probs, axis=1)
        max_probs = np.max(y_probs, axis=1)
        y_pred[max_probs < t] = -1
        valid_idx = (y_pred != -1)

        if np.sum(valid_idx) == 0:
            precision_all.append(0.0)
            for c in classes:
                precision_classes[c].append(0.0)
            continue

        # Overall macro precision
        precision_all.append(precision_score(y_true_arr[valid_idx], y_pred[valid_idx], average='macro', zero_division=0))

        # Per-class precision (one-vs-rest)
        for i, c in enumerate(classes):
            y_true_class = (y_true_arr[valid_idx] == i).astype(int)
            y_pred_class = (y_pred[valid_idx] == i).astype(int)
            p_c = precision_score(y_true_class, y_pred_class, average='binary', zero_division=0)
            precision_classes[c].append(p_c)

    plt.figure()
    plt.plot(thresholds, precision_all, linewidth=4,
             label=f'all classes {np.max(precision_all):.2f} at {thresholds[np.argmax(precision_all)]:.3f}')
    for c in classes:
        plt.plot(thresholds, precision_classes[c], label=c)
    plt.xlabel('Confidence')
    plt.ylabel('Precision')
    plt.title('Precision-Confidence Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_recall_confidence(y_true, y_probs, classes, save_path):
    from sklearn.metrics import recall_score
    thresholds = np.linspace(0, 1, 50)
    recall_all = []
    recall_classes = {cls: [] for cls in classes}
    y_true_arr = np.array(y_true)

    for t in thresholds:
        y_pred = np.argmax(y_probs, axis=1)
        max_probs = np.max(y_probs, axis=1)
        y_pred[max_probs < t] = -1
        valid_idx = (y_pred != -1)

        if np.sum(valid_idx) == 0:
            recall_all.append(0.0)
            for c in classes:
                recall_classes[c].append(0.0)
            continue

        # Overall macro recall
        recall_all.append(recall_score(y_true_arr[valid_idx], y_pred[valid_idx], average='macro', zero_division=0))

        # Per-class recall (one-vs-rest)
        for i, c in enumerate(classes):
            y_true_class = (y_true_arr[valid_idx] == i).astype(int)
            y_pred_class = (y_pred[valid_idx] == i).astype(int)
            r_c = recall_score(y_true_class, y_pred_class, average='binary', zero_division=0)
            recall_classes[c].append(r_c)

    plt.figure()
    plt.plot(thresholds, recall_all, linewidth=4,
             label=f'all classes {np.max(recall_all):.2f} at {thresholds[np.argmax(recall_all)]:.3f}')
    for c in classes:
        plt.plot(thresholds, recall_classes[c], label=c)
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.title('Recall-Confidence Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, classes, save_path):
    """
    Plot Precision-Recall curve for each class and overall mAP for multi-class scenario.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_true_arr = np.array(y_true)
    n_classes = len(classes)
    # Convert y_true to one-hot
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, lbl in enumerate(y_true):
        y_true_onehot[i, lbl] = 1

    plt.figure()
    aps = []
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        aps.append(ap)
        plt.plot(recall, precision, label=f'{c} {ap:.3f}')
    mean_ap = np.mean(aps)
    plt.plot([0,1],[mean_ap,mean_ap], 'k--') # reference line
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (mAP={mean_ap:.3f})')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
