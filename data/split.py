import os
import shutil
import random


def split_dataset(source_dir, dest_dir, train_size=0.7, val_size=0.15):

    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test'), exist_ok=True)

    classes = ['normal', 'benign', 'malignant']

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        print(f"Looking for directory: {cls_dir}")
        if not os.path.exists(cls_dir):
            print(f"Directory does not exist: {cls_dir}")
            continue

        images = os.listdir(cls_dir)
        random.shuffle(images)

        train_end = int(len(images) * train_size)
        val_end = train_end + int(len(images) * val_size)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]


        os.makedirs(os.path.join(dest_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'test', cls), exist_ok=True)


        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(dest_dir, 'train', cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(dest_dir, 'val', cls, img))
        for img in test_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(dest_dir, 'test', cls, img))


source_directory = r'D:\JetBrains\PyCharm Professional\MediPrediction\data'
destination_directory = r'D:\JetBrains\PyCharm Professional\MediPrediction\data'
split_dataset(source_directory, destination_directory)
