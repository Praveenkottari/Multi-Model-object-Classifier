import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models import resnet18, vit
import numpy as np
from plots import (plot_curves, plot_confusion_matrix, plot_sample_predictions,
                   plot_f1_confidence, plot_precision_confidence,
                   plot_recall_confidence, plot_precision_recall_curve)
from sklearn.metrics import accuracy_score
from metrics import save_classification_report, save_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help='dataset directory with train/val')
    parser.add_argument('--model', type=str, default='resnet18', help='model type: resnet18 or vit')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--workers', type=int, default=4, help='num workers for dataloader')
    parser.add_argument('--run_name', type=str, default=None, help='Run name. If None, will generate based on existing runs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create run directory
    if args.run_name is None:
        run_dir = 'results/runs'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        existing_runs = sorted([d for d in os.listdir(run_dir) if d.startswith('run')])
        if len(existing_runs) == 0:
            run_idx = 1
        else:
            last_run = existing_runs[-1]
            run_idx = int(last_run.replace('run','')) + 1
        run_name = f'run{run_idx}'
    else:
        run_name = args.run_name

    run_path = os.path.join('results', 'runs', run_name)
    weights_path = os.path.join(run_path, 'weights')
    os.makedirs(weights_path, exist_ok=True)

    # Data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    class_names = train_dataset.classes

    # Select model

    model_factory = {
        'resnet18': 'models.resnet18',
        'vit': 'models.vit',
        'inception': 'models.inception',
        'vgg16': 'models.vgg16',
        'efficientnet': 'models.efficientnet',
        'densenet': 'models.densenet',
        'mobilenetv2': 'models.mobilenetv2',
        'mobilenetv3': 'models.mobilenetv3',
        'xception': 'models.xception',
        'resnet50': 'models.resnet50',
        'swin': 'models.swin_transformer',
        'deit': 'models.deit'
    }

    chosen_model = args.model
    if chosen_model not in model_factory:
        raise ValueError(f"Unsupported model chosen: {chosen_model}")

    mod = __import__(model_factory[chosen_model], fromlist=['get_model'])
    # For all other models
    model = mod.get_model(args.num_classes)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []


    # Early Stopping Parameters
    early_stop_patience = 10
    min_val_loss_threshold = 0.005  
    best_val_acc = 0.0
    best_val_loss = float('inf') 
    epochs_without_improvement = 0


    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        train_epoch_loss = running_loss / total_train
        train_epoch_acc = running_corrects / total_train
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_epoch_loss = val_running_loss / total_val
        val_epoch_acc = val_running_corrects / total_val
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f} "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Early stopping based on validation loss threshold
        if val_epoch_loss < min_val_loss_threshold:
            print(f"Stopping early as validation loss reached below threshold: {val_epoch_loss:.4f}")
            break

        # Early stopping based on validation accuracy and overfitting
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(weights_path, 'best.pt'))
            epochs_without_improvement = 0  # Reset counter since we got improvement
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping triggered: No improvement in validation accuracy for {early_stop_patience} epochs.")
                break

        # Save best weights
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(weights_path, 'best.pt'))

    # Save last weights
    torch.save(model.state_dict(), os.path.join(weights_path, 'last.pt'))

    # Compute final metrics for val set again but also store probabilities
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []  # to store softmax probabilities
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # raw logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs,1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs)
            
    all_probs = np.concatenate(all_probs, axis=0)

    # Classification report
    save_classification_report(all_labels, all_preds, class_names, run_path)

    # Plot training curves
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, 
                save_path=os.path.join(run_path, 'loss_accuracy.png'))

    # Additional plots
    plot_f1_confidence(all_labels, all_probs, class_names, 
                       save_path=os.path.join(run_path, 'f1_confidence_curve.png'))
    plot_precision_confidence(all_labels, all_probs, class_names, 
                              save_path=os.path.join(run_path, 'precision_confidence_curve.png'))
    plot_recall_confidence(all_labels, all_probs, class_names, 
                           save_path=os.path.join(run_path, 'recall_confidence_curve.png'))
    plot_precision_recall_curve(all_labels, all_probs, class_names, 
                                save_path=os.path.join(run_path, 'precision_recall_curve.png'))

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes=class_names,
                          save_path=os.path.join(run_path, 'confusion_matrix.png'))

    # Plot sample predictions
    inputs, labels = next(iter(val_loader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs,1)
    plot_sample_predictions(inputs.cpu(), labels, preds.cpu(), class_names,
                            save_path=os.path.join(run_path, 'sample_predictions.png'))

    final_val_acc = accuracy_score(all_labels, all_preds)
    print("Training completed!")
    print("******************************************")
    print(f"Final Val Accuracy: {final_val_acc:.4f}")

    final_val_loss = val_losses[-1]
    print(f"Final Val Loss: {final_val_loss:.4f}")

    # Save summary
    train_acc = train_accuracies[-1]
    val_acc = val_accuracies[-1]
    save_summary(args, train_dataset, val_dataset, class_names, final_val_acc, final_val_loss, 
                 train_acc, val_acc, run_path)

if __name__ == '__main__':
    main()
