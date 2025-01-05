import argparse
import os
import yaml
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


def load_yaml_classes(yaml_path):
    """
    Loads class names from a YAML file that has a structure like:
    
    classes:
      - classA
      - classB
      - ...
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # We assume 'classes' key in the YAML
    return data['classes']


def load_model(model_name, num_classes, weights_path, device='cpu'):
    """
    Loads the chosen model with the specified num_classes and
    loads the state dict from weights_path.
    """
    # Example model factory that maps model names to actual Python modules
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

    if model_name not in model_factory:
        raise ValueError(f"Model '{model_name}' is not implemented in model_factory.")

    # Dynamically import the module that contains the get_model() function
    mod = __import__(model_factory[model_name], fromlist=['get_model'])
    # Retrieve the function
    model_fn = getattr(mod, 'get_model')
    
    # Create model
    model = model_fn(num_classes=num_classes)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_image(img_path, transform):
    """
    Load an image (RGB) and apply the given transform.
    Return a 4D tensor (batch=1).
    """
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)


def predict_single_image(img_path, model, device, transform, class_names):
    """
    Predict class for a single image.
    Returns (pred_index, pred_class_name, pred_score).
    """
    input_tensor = prepare_image(img_path, transform).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        # Convert logits to probabilities via softmax
        probs = torch.softmax(output, dim=1)
        top_prob, pred = torch.max(probs, dim=1)
    
    pred_idx = pred.item()
    pred_class = class_names[pred_idx]
    score = top_prob.item()  # probability of the predicted class
    return pred_idx, pred_class, score


def plot_prediction(img_path, predicted_class, score, save_path):
    """
    Plot the image with the predicted class (and score) as title and save to disk.
    """
    img = Image.open(img_path).convert('RGB')
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {predicted_class} (score={score:.2f})")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Folder containing images to predict (if not using --single_image).")
    parser.add_argument('--single_image', type=str, default=None,
                        help="Path to a single image to predict. If provided, data_dir is ignored.")
    parser.add_argument('--model', type=str, default='resnet18', help='Model name (e.g., resnet18, vit).')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights (.pth file).')
    parser.add_argument('--yaml', type=str, required=True, help='Path to YAML file with class names.')
    parser.add_argument('--run_name', type=str, default=None, help='Name for this prediction run (folder).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    args = parser.parse_args()

    # Resolve device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load class names from YAML
    class_names = load_yaml_classes(args.yaml)
    num_classes = len(class_names)

    # Create base directory for predictions
    predict_base = os.path.join('results', 'predict')
    os.makedirs(predict_base, exist_ok=True)

    # If no run_name is provided, auto-increment run folders: run1, run2, ...
    if args.run_name is None:
        existing_runs = [d for d in os.listdir(predict_base) if d.startswith('run')]
        run_ids = []
        for run in existing_runs:
            try:
                run_id = int(run.replace('run', ''))
                run_ids.append(run_id)
            except ValueError:
                pass
        current_run_id = max(run_ids) + 1 if run_ids else 1
        run_folder = os.path.join(predict_base, f'run{current_run_id}')
    else:
        run_folder = os.path.join(predict_base, args.run_name)

    os.makedirs(run_folder, exist_ok=True)
    
    # Load model
    model = load_model(args.model, num_classes, args.weights, device)

    # Define transform (same as training/validation typically)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    sample_paths = []
    predictions = []
    scores = []

    # Single image mode
    if args.single_image is not None:
        img_path = args.single_image
        _, pred_class, score = predict_single_image(img_path, model, device, transform, class_names)
        sample_paths.append(img_path)
        predictions.append(pred_class)
        scores.append(score)

        # Plot single image
        out_img_path = os.path.join(run_folder, 'single_prediction.png')
        plot_prediction(img_path, pred_class, score, out_img_path)

    else:
        # Folder mode
        if not args.data_dir:
            raise ValueError("Please provide either --single_image or --data_dir with images.")
        
        # List all image files in data_dir
        image_files = sorted([
            f for f in os.listdir(args.data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ])
        
        for img_file in image_files:
            img_path = os.path.join(args.data_dir, img_file)
            _, pred_class, score = predict_single_image(img_path, model, device, transform, class_names)
            sample_paths.append(img_path)
            predictions.append(pred_class)
            scores.append(score)
        
        # Plot a few sample predictions (e.g., the first 5)
        for i in range(min(5, len(sample_paths))):
            sample_img = sample_paths[i]
            sample_pred = predictions[i]
            sample_score = scores[i]
            out_img_path = os.path.join(run_folder, f'predicted_{i+1}.png')
            plot_prediction(sample_img, sample_pred, sample_score, out_img_path)

    # Save results (image_id, predicted_class, score) in CSV
    data_rows = []
    for path, pred_class, score in zip(sample_paths, predictions, scores):
        data_rows.append({
            'image_id': os.path.basename(path),
            'predicted_class': pred_class,
            'score': f"{score:.4f}"
        })
    df = pd.DataFrame(data_rows)
    
    csv_path = os.path.join(run_folder, 'predictions.csv')
    df.to_csv(csv_path, index=False)

    print(f"\nPrediction run saved to: {run_folder}")
    print(f"Total images processed: {len(sample_paths)}")
    if len(sample_paths) > 0:
        print(f"First image predicted as '{predictions[0]}' with score={scores[0]:.2f}")


if __name__ == '__main__':
    main()
