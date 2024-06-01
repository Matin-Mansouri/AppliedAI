import os
import random
import shutil
from PIL import Image

# Paths to data
base_path = '/content/drive/MyDrive/ColabNotebooks/DataSet'
preprocessed_path = '/content/drive/MyDrive/ColabNotebooks/DataSet'

# Classes
classes = ['airport_terminal','market','images_flattened','museum','restaurant']

# Ensure the preprocessed data directory exists
if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)

# Split ratios
train_ratio = 0.64
val_ratio = 0.16
test_ratio = 0.20

for class_name in classes:
    class_path = os.path.join(base_path, class_name)
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    # Randomly sample 500 images
    sampled_images = random.sample(images, 500)

    # Calculate the number of images for each split
    train_size = int(train_ratio * len(sampled_images))
    val_size = int(val_ratio * len(sampled_images))
    test_size = len(sampled_images) - train_size - val_size

    # Shuffle the images to ensure randomness
    random.shuffle(sampled_images)

    # Split the images
    train_images = sampled_images[:train_size]
    val_images = sampled_images[train_size:train_size + val_size]
    test_images = sampled_images[train_size + val_size:]

    # Create directories for the splits
    splits = ['train', 'val', 'test']
    for split in splits:
        split_class_path = os.path.join(preprocessed_path, split, class_name)
        if not os.path.exists(split_class_path):
            os.makedirs(split_class_path)

    # Preprocess and copy images to the respective directories
    for image in train_images:
        src_image_path = os.path.join(class_path, image)
        dst_image_path = os.path.join(preprocessed_path, 'train', class_name, image)
        with Image.open(src_image_path) as img:
            img = img.resize((256, 256))
            img.save(dst_image_path)

    for image in val_images:
        src_image_path = os.path.join(class_path, image)
        dst_image_path = os.path.join(preprocessed_path, 'val', class_name, image)
        with Image.open(src_image_path) as img:
            img = img.resize((256, 256))
            img.save(dst_image_path)

    for image in test_images:
        src_image_path = os.path.join(class_path, image)
        dst_image_path = os.path.join(preprocessed_path, 'test', class_name, image)
        with Image.open(src_image_path) as img:
            img = img.resize((256, 256))
            img.save(dst_image_path)

print("Data preprocessing and splitting completed successfully.")
