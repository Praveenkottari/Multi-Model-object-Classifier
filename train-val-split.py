import os
import shutil
import random
import argparse
from math import ceil

def split_dataset(dataset_dir, train_dir, val_dir, train_ratio=0.75, val_ratio=0.25):
    """
    Splits a dataset into train and validation sets.

    Args:
    - dataset_dir (str): Directory containing the dataset.
    - train_dir (str): Directory to store training data.
    - val_dir (str): Directory to store validation data.
    - train_ratio (float): The ratio of the dataset to be used for training.
    - val_ratio (float): The ratio of the dataset to be used for validation.
    """
    # Ensure the output directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Iterate over each class in the dataset directory
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if os.path.isdir(class_dir):
            # Ensure class directories exist in train and val directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            
            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            if not os.path.exists(val_class_dir):
                os.makedirs(val_class_dir)

            # Get all samples in the class directory
            samples = os.listdir(class_dir)
            random.shuffle(samples)

            # Split the samples
            split_index1 = ceil(len(samples) * train_ratio)
            split_index2 = ceil(len(samples) * (train_ratio + val_ratio))

            train_samples = samples[:split_index1]
            val_samples = samples[split_index1:split_index2]

            # Copy the samples to the respective directories
            for sample in train_samples:
                src = os.path.join(class_dir, sample)
                dst = os.path.join(train_class_dir, sample)
                shutil.copy(src, dst)

            for sample in val_samples:
                src = os.path.join(class_dir, sample)
                dst = os.path.join(val_class_dir, sample)
                shutil.copy(src, dst)

            print(f'Class "{class_name}": {len(train_samples)} train samples, {len(val_samples)} val samples')

def main():
    # Set up argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the output training data directory.")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the output validation data directory.")
    parser.add_argument('--train_ratio', type=float, default=0.75, help="The ratio of the dataset to be used for training (default: 0.75).")
    parser.add_argument('--val_ratio', type=float, default=0.25, help="The ratio of the dataset to be used for validation (default: 0.25).")

    # Parse the arguments
    args = parser.parse_args()

    # Validate ratios add up to 1
    if args.train_ratio + args.val_ratio != 1.0:
        print("Error: The sum of train_ratio and val_ratio must be 1.0.")
        return

    # Split the dataset
    split_dataset(args.dataset_dir, args.train_dir, args.val_dir, args.train_ratio, args.val_ratio)
    print("Train-Validation split completed")

if __name__ == "__main__":
    main()
