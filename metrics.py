from sklearn.metrics import classification_report
import os
import datetime


def save_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write(report)

def save_summary(args, train_dataset, val_dataset, class_names, final_val_acc, final_val_loss, 
                 train_acc, val_acc, save_path):
    """
    Save a summary text file with training configuration and final results.
    """

    # Get the current date and time
    current_time = datetime.datetime.now()
    # Format the current time as a string
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    with open(os.path.join(save_path, 'training_summary.txt'), 'w') as f:
        f.write("Training Summary\n")
        f.write(f"Program executed at: {formatted_time}\n")
        f.write("================\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of classes: {args.num_classes}\n")
        f.write(f"Class names: {', '.join(class_names)}\n")
        f.write(f"Number of Training samples: {num_train_samples}\n")
        f.write(f"Number of Validation samples: {num_val_samples}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Number of workers: {args.workers}\n")
        f.write("\nFinal Results\n")
        f.write("=============\n")
        f.write(f"Final Train Accuracy: {train_acc*100:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_acc*100:.2f}%\n")
        f.write(f"Final Validation Loss: {final_val_loss:.4f}\n")
        f.write(f"Final Validation Accuracy (recomputed): {final_val_acc*100:.2f}%\n")


