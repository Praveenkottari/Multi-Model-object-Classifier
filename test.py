import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.test_plot import plot_confusion_matrix
from utils.test_reports import save_classification_report, save_predictions_csv, save_summary
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help='dataset directory with test folder')
    parser.add_argument('--model', type=str, default='resnet18', help='model type (same as used during training)')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='num workers for dataloader')
    parser.add_argument('--weights', type=str, required=True, help='path to the trained weights .pt file')
    parser.add_argument('--device', type=str, default='cuda', help='device to use (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create inference base directory
    base_inference_dir = os.path.join('results', 'inference')
    os.makedirs(base_inference_dir, exist_ok=True)

    # Find existing test runs and create a new one
    existing_runs = [d for d in os.listdir(base_inference_dir) if d.startswith('test')]
    run_ids = []
    for run in existing_runs:
        # run should be in format testX where X is an integer
        try:
            run_id = int(run.replace('test', ''))
            run_ids.append(run_id)
        except ValueError:
            pass
    if len(run_ids) == 0:
        current_run_id = 1
    else:
        current_run_id = max(run_ids) + 1

    inference_dir = os.path.join(base_inference_dir, f'test{current_run_id}')
    os.makedirs(inference_dir, exist_ok=True)

    # Data transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    class_names = test_dataset.classes

    # Import and create model
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
        'deit': 'models.deit',
        'SimpleConvNet':'models.SimpleConvNet',
        'SimpleViTNet' : 'models.SimpleViTNet'
    }

    chosen_model = args.model
    if chosen_model not in model_factory:
        raise ValueError(f"Unsupported model chosen: {chosen_model}")

    mod = __import__(model_factory[chosen_model], fromlist=['get_model'])
    model = mod.get_model(args.num_classes)

    # Load weights
    state_dict = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    image_paths = []

    batch_num = 1
    with torch.no_grad():
        for inputs, labels in test_loader:
            print("Testting image sample batch :", batch_num)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            batch_num+=1

    for sample in test_dataset.samples:
        image_paths.append(sample[0])

    # Compute accuracy
    print("Testing completed!")
    print("**********************************************")
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Overall Test Accuracy: {accuracy*100:.2f}%")

    # Save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes=class_names, save_path=os.path.join(inference_dir, 'confusion_matrix.png'))

    # Save classification report
    save_classification_report(all_labels, all_preds, class_names, inference_dir)

    # Save predictions csv
    save_predictions_csv(image_paths, all_labels, all_preds, class_names, inference_dir)

    # Save summary (includes classwise accuracy, etc.)
    save_summary(class_names, all_labels, all_preds, args.model, len(test_dataset), inference_dir)

if __name__ == '__main__':
    main()
