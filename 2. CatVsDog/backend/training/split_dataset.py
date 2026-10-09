import os
import random
import shutil


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# Folder containing the original images
SOURCE_FOLDER = "../../data/raw"

# Destination folders
TRAIN_FOLDER = "../../data/train"
VAL_FOLDER = "../../data/val"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# 80% training, 20% validation
TRAIN_RATIO = 0.8


# ---------------------------------------------------------
# Create required folders
# ---------------------------------------------------------

os.makedirs(f"{TRAIN_FOLDER}/cats", exist_ok=True)
os.makedirs(f"{TRAIN_FOLDER}/dogs", exist_ok=True)

os.makedirs(f"{VAL_FOLDER}/cats", exist_ok=True)
os.makedirs(f"{VAL_FOLDER}/dogs", exist_ok=True)


# ---------------------------------------------------------
# Get all image file names
# ---------------------------------------------------------

all_files = os.listdir(SOURCE_FOLDER)

cats = []
dogs = []


# Separate cat and dog images based on filename
for filename in all_files:

    if filename.startswith("cat."):
        cats.append(filename)

    elif filename.startswith("dog."):
        dogs.append(filename)


print("Total cats:", len(cats))
print("Total dogs:", len(dogs))


# ---------------------------------------------------------
# Shuffle images
# ---------------------------------------------------------

# We shuffle so training/validation samples are random
random.shuffle(cats)
random.shuffle(dogs)


# ---------------------------------------------------------
# Function to split files
# ---------------------------------------------------------

def split_images(files, animal_name):

    # Calculate position where training set ends
    split_index = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_index]
    val_files = files[split_index:]

    print(
        animal_name,
        "training:",
        len(train_files),
        "validation:",
        len(val_files)
    )

    # Copy training images
    for filename in train_files:

        source = os.path.join(SOURCE_FOLDER, filename)

        destination = os.path.join(
            TRAIN_FOLDER,
            animal_name,
            filename
        )

        shutil.copy(source, destination)

    # Copy validation images
    for filename in val_files:

        source = os.path.join(SOURCE_FOLDER, filename)

        destination = os.path.join(
            VAL_FOLDER,
            animal_name,
            filename
        )

        shutil.copy(source, destination)


# ---------------------------------------------------------
# Run split
# ---------------------------------------------------------

split_images(cats, "cats")
split_images(dogs, "dogs")

print("Dataset split completed!")