import pandas as pd
import os


def organize_ham10000(images_dir, metadata_path, output_dir):
    """
    Organize HAM10000 dataset into train/test directories based on metadata
    """
    # Read metadata
    metadata = pd.read_csv(metadata_path)

    # Create output directories
    for split in ['train', 'test']:
        for class_name in CLASSES:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # Split data (using lesion_id as grouping to avoid data leakage)
    unique_lesions = metadata['lesion_id'].unique()
    train_lesions = set(pd.Series(unique_lesions).sample(frac=0.8))  # 80% train

    # Organize images
    for _, row in metadata.iterrows():
        src_path = os.path.join(images_dir, row['image_id'] + '.jpg')
        if not os.path.exists(src_path):
            continue

        # Determine split
        split = 'train' if row['lesion_id'] in train_lesions else 'test'

        # Destination path
        dest_dir = os.path.join(output_dir, split, row['dx'])
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, row['image_id'] + '.jpg')

        # Copy or symlink the image
        if not os.path.exists(dest_path):
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy(src_path, dest_path)
            else:  # Unix
                os.symlink(os.path.abspath(src_path), dest_path)

    print("Dataset organization complete!")