import h5py
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype= np.uint8)
    return img_array

def create_hdf5_dataset(image_folder, hdf5_file, target_size=(224, 224)):
    classes = ['normal', 'pneumonia']
    image_paths = []
    labels = []

    for label, class_name in enumerate(classes):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                if image_name.lower().endswith('.jpeg'):
                    image_paths.append(os.path.join(class_folder, image_name))
                    labels.append(label)

    num_images = len(image_paths)

    with h5py.File(hdf5_file, 'w') as hf:
        img_shape = (num_images, target_size[0], target_size[1])
        hf.create_dataset('images', shape=img_shape, dtype='uint8')
        hf.create_dataset('labels', shape=(num_images,), dtype='uint8')

        for i, image_path in enumerate(image_paths):
            img_array = preprocess_image(image_path, target_size)
            hf['images'][i] = img_array
            hf['labels'][i] = labels[i]

            if i % 100 == 0:
                print(f"Processed {i} of {num_images} images.")

    print(f"All images have been saved to {hdf5_file}")

image_folder = 'dataset'
hdf5_file = 'x_ray_full_dataset.h5'
create_hdf5_dataset(image_folder, hdf5_file, target_size=(180, 180))