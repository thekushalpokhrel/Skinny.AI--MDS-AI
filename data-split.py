import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
DATASET_ROOT = 'dataverse_files'
IMAGE_DIRS = [
    os.path.join(DATASET_ROOT, 'HAM10000_images_part_1'),
    os.path.join(DATASET_ROOT, 'HAM10000_images_part_2')
]
METADATA_FILE = os.path.join(DATASET_ROOT, 'HAM10000_metadata.csv')
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
TEST_SIZE = 0.3  # 30% for test, 70% for train
RANDOM_STATE = 42  # For reproducible splits


def setup_folders():
    """Create train/test folders with class subdirectories"""
    for folder in [TRAIN_DIR, TEST_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)
        for class_name in CLASSES:
            Path(os.path.join(folder, class_name)).mkdir(exist_ok=True)


def find_image_path(image_id):
    """Search for image in both part directories"""
    for img_dir in IMAGE_DIRS:
        img_path = os.path.join(img_dir, f'{image_id}.jpg')
        if os.path.exists(img_path):
            return img_path
    return None


def organize_images():
    """Organize images into train/test folders with 70-30 split"""
    df = pd.read_csv(METADATA_FILE)

    for class_name in CLASSES:
        print(f"\nProcessing {class_name}...")
        class_df = df[df['dx'] == class_name]

        # Get available images
        available_images = []
        for _, row in class_df.iterrows():
            img_path = find_image_path(row['image_id'])
            if img_path:
                available_images.append(img_path)

        if not available_images:
            print(f"No images found for {class_name}, skipping...")
            continue

        # Split and copy
        train_files, test_files = train_test_split(
            available_images,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # Copy training images
        for img_path in train_files:
            dest = os.path.join(TRAIN_DIR, class_name, os.path.basename(img_path))
            shutil.copy2(img_path, dest)  # copy2 preserves metadata

        # Copy test images
        for img_path in test_files:
            dest = os.path.join(TEST_DIR, class_name, os.path.basename(img_path))
            shutil.copy2(img_path, dest)

        print(f"  Total images: {len(available_images)}")
        print(f"  Training set: {len(train_files)} images")
        print(f"  Test set: {len(test_files)} images")


def main():
    print("Starting dataset organization...")
    setup_folders()
    organize_images()

    # Final summary
    print("\n=== Dataset organization complete ===")
    print(f"\nTraining set ({TRAIN_DIR}):")
    for class_name in CLASSES:
        count = len(os.listdir(os.path.join(TRAIN_DIR, class_name)))
        print(f"  {class_name}: {count} images")

    print(f"\nTest set ({TEST_DIR}):")
    for class_name in CLASSES:
        count = len(os.listdir(os.path.join(TEST_DIR, class_name)))
        print(f"  {class_name}: {count} images")


if __name__ == '__main__':
    main()