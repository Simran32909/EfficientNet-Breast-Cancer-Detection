import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, split_ratios=(0.7, 0.15, 0.15)):
    """
    Splits a dataset of images into training, validation, and test sets.

    The source directory should have subdirectories for each class, e.g.:
    source_dir/
    ...class_a/
    ......img1.png
    ......img2.png
    ...class_b/
    ......img1.png

    The destination directory will be created with train, val, and test subdirectories.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    train_ratio, val_ratio, test_ratio = split_ratios
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

    # Create destination directories
    train_path = os.path.join(dest_dir, 'train')
    val_path = os.path.join(dest_dir, 'val')
    test_path = os.path.join(dest_dir, 'test')

    for path in [train_path, val_path, test_path]:
        os.makedirs(path, exist_ok=True)

    print(f"Splitting dataset from '{source_dir}' into '{dest_dir}'...")

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create class subdirectories in train, val, test
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

        # Get list of all images and shuffle them
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(images)

        # Calculate split indices
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        # Get file lists for each set
        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        # Copy files to destination
        for file_list, dest_path in [(train_files, train_path), (val_files, val_path), (test_files, test_path)]:
            for filename in tqdm(file_list, desc=f"Copying {class_name} to {os.path.basename(dest_path)}"):
                shutil.copy(os.path.join(class_dir, filename), os.path.join(dest_path, class_name, filename))
    
    print("\nDataset splitting complete!")

if __name__ == '__main__':
    SOURCE_DIRECTORY = 'dataset'
    DESTINATION_DIRECTORY = 'data_processed'
    split_dataset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY) 