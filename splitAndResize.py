import os
import random
import shutil
from tqdm import tqdm
from PIL import Image

# funkcija ki spremeni velikost slike na 64x64
def resize_image(image_path, output_path, size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path)

# funkcija, ki preveri, če so vse slike 64x64
def verify_image_sizes(directory_path, expected_size=(64, 64)):
    mismatched_sizes = {}
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.ppm'):
                image_path = os.path.join(root, filename)
                with Image.open(image_path) as img:
                    if img.size != expected_size:
                        mismatched_sizes[filename] = img.size
    return mismatched_sizes

# funkcija, ki gre skozi vse direktorije (razrede) in randomly vzame x število YYYY (augmentiranih) slik za vsako XXXX sliko 
def split_and_copy_dataset(base_path, output_base, train_ratio=0.8):
    random.seed(42)

    # ustvarjanje direktorijev za train in validation
    train_path = os.path.join(output_base, 'train')
    validation_path = os.path.join(output_base, 'validation')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)

    for class_id in tqdm(sorted(os.listdir(base_path)), desc='Processing classes'):
        class_dir = os.path.join(base_path, class_id)
        if not os.path.isdir(class_dir):
            continue

        # ustvarjanje poddirektorijev v train in validation
        train_class_path = os.path.join(train_path, class_id)
        validation_class_path = os.path.join(validation_path, class_id)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(validation_class_path, exist_ok=True)

        # zbiranj slik
        images_by_base = {}
        for filename in os.listdir(class_dir):
            if filename.endswith('.ppm'):
                base_image, _ = filename.split('_', 1)
                if base_image in images_by_base:
                    images_by_base[base_image].append(filename)
                else:
                    images_by_base[base_image] = [filename]

        # Split and copy files for each base image
        for base_image, filenames in tqdm(images_by_base.items(), desc=f'Class {class_id}'):
            random.shuffle(filenames)
            split_index = int(len(filenames) * train_ratio)
            training_files = filenames[:split_index]
            validation_files = filenames[split_index:]

            # resize in copy
            for file in training_files:
                src_file_path = os.path.join(class_dir, file)
                dest_file_path = os.path.join(train_class_path, file)
                try:
                    resize_image(src_file_path, dest_file_path)
                except Exception as e:
                    print(f"Error resizing/copying {src_file_path} to {dest_file_path}: {e}")
            for file in validation_files:
                src_file_path = os.path.join(class_dir, file)
                dest_file_path = os.path.join(validation_class_path, file)
                try:
                    resize_image(src_file_path, dest_file_path)
                except Exception as e:
                    print(f"Error resizing/copying {src_file_path} to {dest_file_path}: {e}")

# uporaba
base_path = 'Images'
output_base = 'DividedImages'
split_and_copy_dataset(base_path, output_base)

mismatched_sizes = verify_image_sizes(output_base)
if not mismatched_sizes:
    print("All images are 64x64.")
else:
    print("Some images are not 64x64:")
    for filename, size in mismatched_sizes.items():
        print(f"{filename}: {size}")