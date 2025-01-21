import os
import shutil
import random

# Rutas principales
DATA_ALL_DIR = './data_all'  # Carpeta con todas las imágenes originales
DATA_OUT_DIR = './data'  # Carpeta donde se guardará el conjunto de datos YOLO

def split_dataset(data_all_dir, output_dir, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    # Check ratios sum to 1
    if train_ratio + test_ratio + val_ratio != 1.0:
        raise ValueError("The sum of train_ratio, test_ratio, and val_ratio must be 1.")

    # Define input directories
    images_dir = os.path.join(data_all_dir, 'images')
    labels_dir = os.path.join(data_all_dir, 'labels')

    # Create output directories
    for subset in ['train', 'test', 'validation']:
        os.makedirs(os.path.join(output_dir, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', subset), exist_ok=True)

    # Get list of image files and corresponding label files
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # Ensure that each image has a corresponding label
    images.sort()
    labels.sort()

    paired_files = [(img, img.replace(os.path.splitext(img)[1], '.txt')) for img in images if img.replace(os.path.splitext(img)[1], '.txt') in labels]

    if not paired_files:
        raise ValueError("No matching images and labels found.")

    # Shuffle the data
    random.shuffle(paired_files)

    # Split the data
    total_files = len(paired_files)
    train_end = int(total_files * train_ratio)
    test_end = train_end + int(total_files * test_ratio)

    train_files = paired_files[:train_end]
    test_files = paired_files[train_end:test_end]
    val_files = paired_files[test_end:]

    # Helper function to copy files
    def copy_files(file_pairs, subset):
        for img_file, label_file in file_pairs:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(output_dir, 'images', subset, img_file))
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, 'labels', subset, label_file))

    # Copy the files
    copy_files(train_files, 'train')
    copy_files(test_files, 'test')
    copy_files(val_files, 'validation')

    # Verify output
    for subset in ['train', 'test', 'validation']:
        image_count = len(os.listdir(os.path.join(output_dir, 'images', subset)))
        label_count = len(os.listdir(os.path.join(output_dir, 'labels', subset)))
        print(f"{subset.capitalize()} set: {image_count} images, {label_count} labels")

        if image_count == 0 or label_count == 0:
            raise RuntimeError(f"The {subset} subset is empty. Check the data splitting ratios and input data.")

    print(f"Dataset successfully split into train, test, and validation subsets.")

# Llamada al script con las rutas principales
def main():
    split_dataset(DATA_ALL_DIR, DATA_OUT_DIR)

if __name__ == "__main__":
    main()