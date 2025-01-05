import os
import random
import shutil

def move_random_images(src_folder, dest_folder, percentage=10):
    """
    Moves a percentage of random images from src_folder to dest_folder.

    Args:
    src_folder (str): Path to the source folder containing images.
    dest_folder (str): Path to the destination folder.
    percentage (int): Percentage of images to move (default is 10).
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get a list of all image files in the source folder
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # Calculate the number of files to move
    num_files_to_move = max(1, int(len(all_files) * percentage / 100))
    
    # Randomly select files to move
    files_to_move = random.sample(all_files, num_files_to_move)
    
    # Move the selected files
    for file_name in files_to_move:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.move(src_path, dest_path)
    
    print(f"Moved {num_files_to_move} images from '{src_folder}' to '{dest_folder}'.")

# Example usage
source_folder = "./dataset/val/background"
destination_folder = "./dataset/test/background"
move_random_images(source_folder, destination_folder, percentage=10)
