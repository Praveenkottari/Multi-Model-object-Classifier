import os
import csv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import datetime

def save_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write(report)

def save_predictions_csv(image_paths, y_true, y_pred, classes, save_path):
    # Write a CSV file with image_id, original_class, predicted_class
    csv_path = os.path.join(save_path, 'predictions.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'original_class', 'predicted_class'])
        for img_path, true_label, pred_label in zip(image_paths, y_true, y_pred):
            image_id = os.path.basename(img_path)
            writer.writerow([image_id, classes[true_label], classes[pred_label]])

def save_summary(classes, y_true, y_pred, model_name, num_test_samples, save_path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Get the current date and time
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # Classwise accuracy
    classwise_acc = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for i, c in enumerate(classes):
        idx = (y_true_arr == i)
        if np.sum(idx) > 0:
            class_acc = np.sum(y_pred_arr[idx] == i) / np.sum(idx)
        else:
            class_acc = 0.0
        classwise_acc.append((c, class_acc))

    with open(os.path.join(save_path, 'test_summary.txt'), 'w') as f:
        f.write("Test Summary\n")
        f.write(f"Program executed at: {formatted_time}\n")
        f.write("============\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Class names: {', '.join(classes)}\n")
        f.write(f"Testing samples: {num_test_samples}\n")
        f.write("\nFinal Results\n")
        f.write("=============\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"Precision (macro): {prec*100:.2f}%\n")
        f.write(f"Recall (macro): {rec*100:.2f}%\n")
        f.write(f"F1-score (macro): {f1*100:.2f}%\n\n")

        f.write("Classwise Accuracy:\n")
        for c, ca in classwise_acc:
            f.write(f"{c}: {ca*100:.2f}%\n")
