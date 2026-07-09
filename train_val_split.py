# Adapted from: "train_val_split.py" by Evan EdjeElectronics (2025)
# Availability: https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py
# Adjustment Date: 07/08/2026

# Split between train and val folders

from pathlib import Path
import random
import os
import sys
import shutil
import argparse

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".8")',
                    default=.8)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# Check for valid entries
if not os.path.isdir(data_path):
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(0)
if train_percent < .01 or train_percent > 0.99:
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(0)
val_percent = 1 - train_percent

# Define path to input dataset 
input_image_path = os.path.join(data_path,'images')
input_label_path = os.path.join(data_path,'labels')

# Define paths to image and annotation folders
cwd = os.getcwd()
train_img_path = os.path.join(cwd,'data/train/images')
train_txt_path = os.path.join(cwd,'data/train/labels')
val_img_path = os.path.join(cwd,'data/validation/images')
val_txt_path = os.path.join(cwd,'data/validation/labels')

# Create folders if they don't already exist
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Created folder at {dir_path}.')


# Get list of all images and annotation files

""" Adjustment code to prevent:
    - 'validation' folder empty
    - overlapping loops (from O(N^2) -> O(N))
    - duplicate processing of images and labels
    Ensure an image goes to either train or validation. """

# Only grab actual image files (prevent hidden system files like .DS_Store that cause issues)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

img_file_list = [
    path for path in Path(input_image_path).rglob('*') 
    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
]

random.shuffle(img_file_list)  # Shuffle the list randomly once

file_num = len(img_file_list)
train_num = int(file_num * train_percent)

train_files = img_file_list[:train_num]
val_files = img_file_list[train_num:]

print(f'Total images found: {file_num}')
print(f'Images moving to train: {len(train_files)}')
print(f'Images moving to validation: {len(val_files)}')

# Helper function to safely copy images and matching labels
def copy_dataset_split(file_list, target_img_dir, target_lbl_dir):
    for img_path in file_list:
        img_fn = img_path.name
        base_fn = img_path.stem
        txt_fn = base_fn + '.txt'
        txt_path = os.path.join(input_label_path, txt_fn)
        
        # Copy Image
        shutil.copy(img_path, os.path.join(target_img_dir, img_fn))
        
        # Copy Label if it exists (handles background images)
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(target_lbl_dir, txt_fn))

print("Copying training files...")
copy_dataset_split(train_files, train_img_path, train_txt_path)

print("Copying validation files...")
copy_dataset_split(val_files, val_img_path, val_txt_path)

print("Dataset split complete!")