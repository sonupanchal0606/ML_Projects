import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------
# 1. Select device
# ---------------------------------------------------------

# Apple Silicon Macs may support Apple's Metal GPU (MPS).
# Otherwise PyTorch will use the CPU.

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ---------------------------------------------------------
# 2. Image transformations
# ---------------------------------------------------------

# Neural networks require all images to have the same size.
# We'll resize everything to 128x128.
#
# ToTensor converts image pixels from:
#
# 0 - 255
#
# into:
#
# 0.0 - 1.0

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# - Compose(...) runs the listed transformations in order.
# - Resize((128, 128)) resizes every image to 128 pixels tall and 128 pixels wide.
# - ToTensor() converts the image into a PyTorch tensor. For a typical RGB image with 8-bit pixel values, 
#   it also scales values from 0–255 to 0–1 and arranges dimensions as channels × height × width.
# The resulting RGB tensor has shape [3, 128, 128], where 3 represents red, green, and blue channels.


# ---------------------------------------------------------
# 3. Load datasets
# ---------------------------------------------------------

train_dataset = datasets.ImageFolder(
    "../../data/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "../../data/val",
    transform=transform
)


print("Classes:", train_dataset.classes)

# Usually ImageFolder will assign:
#
# cats = 0
# dogs = 1


# ---------------------------------------------------------
# 4. Create DataLoaders
# ---------------------------------------------------------

# Batch size 32 means:
#
# Instead of processing one image at a time,
# process 32 images together.

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ---------------------------------------------------------
# 5. Define CNN
# ---------------------------------------------------------

class CatDogCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------
        # Feature extraction
        # ---------------------------------------------

        self.features = nn.Sequential(

            # Input:
            # 3 x 128 x 128
            #
            # 3 = RGB channels

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # Reduce:
            # 128x128 -> 64x64
            nn.MaxPool2d(2),


            # Second convolution layer
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 64x64 -> 32x32
            nn.MaxPool2d(2),


            # Third convolution layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 32x32 -> 16x16
            nn.MaxPool2d(2)
        )


        # ---------------------------------------------
        # Classification layer
        # ---------------------------------------------

        self.classifier = nn.Sequential(

            # Flatten:
            #
            # 64 feature maps
            # each 16 x 16

            nn.Flatten(),

            nn.Linear(
                64 * 16 * 16,
                128
            ),

            nn.ReLU(),

            # Output 2 values:
            #
            # Cat
            # Dog
            nn.Linear(
                128,
                2
            )
        )


    def forward(self, x):

        # Extract image features
        x = self.features(x)

        # Classify those features
        x = self.classifier(x)

        return x


# ---------------------------------------------------------
# 6. Create model
# ---------------------------------------------------------

model = CatDogCNN()

model = model.to(device)


# ---------------------------------------------------------
# 7. Loss function
# ---------------------------------------------------------

# CrossEntropyLoss is commonly used for classification.

criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------
# 8. Optimizer
# ---------------------------------------------------------

# Adam changes the neural network weights
# while training.

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ---------------------------------------------------------
# 9. Training configuration
# ---------------------------------------------------------

epochs = 10


# ---------------------------------------------------------
# 10. Training loop
# ---------------------------------------------------------

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        # Move data to CPU / MPS
        images = images.to(device)
        labels = labels.to(device)


        # Reset gradients from previous iteration
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate error
        loss = criterion(outputs, labels)


        # Calculate gradients
        loss.backward()


        # Update model weights
        optimizer.step()


        # Add loss
        running_loss += loss.item()


        # Get predicted class
        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    training_accuracy = (
        100 * correct / total
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0


    # We don't need gradients while validating.
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    validation_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {training_accuracy:.2f}% "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )


# ---------------------------------------------------------
# 11. Save trained model
# ---------------------------------------------------------

import os

os.makedirs("../model", exist_ok=True)

torch.save(
    model.state_dict(),
    "../model/cat_dog_model.pth"
)

print("Model saved!")