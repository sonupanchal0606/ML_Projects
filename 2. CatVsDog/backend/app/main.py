import io

import torch
import torch.nn as nn

from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from torchvision import transforms


# ---------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Cat vs Dog Classifier API"
)


# ---------------------------------------------------------
# Allow React frontend to access backend
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    # React Vite normally runs here
    allow_origins=[
        "http://localhost:5174"
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ---------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------

# This architecture MUST be the same architecture
# that we used during training.

class CatDogCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 16 * 16,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = CatDogCNN()

model.load_state_dict(
    torch.load(
        "../model/cat_dog_model.pth",
        map_location=device
    )
)

model = model.to(device)

# Prediction mode
model.eval()


# ---------------------------------------------------------
# Image transformation
# ---------------------------------------------------------

transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor()

])


# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------

classes = [
    "Cat",
    "Dog"
]


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Cat vs Dog Classifier API is running!"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded file
    image_bytes = await file.read()


    # Convert bytes into image
    image = Image.open(
        io.BytesIO(image_bytes)
    )


    # Make sure image is RGB
    image = image.convert("RGB")


    # Transform image
    image_tensor = transform(image)


    # Add batch dimension
    #
    # Before:
    # [3, 128, 128]
    #
    # After:
    # [1, 3, 128, 128]

    image_tensor = image_tensor.unsqueeze(0)


    # Send to CPU / MPS
    image_tensor = image_tensor.to(device)


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(image_tensor)


        # Convert raw neural network values
        # into probabilities.
        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities,
            1
        )


    predicted_class = classes[
        predicted.item()
    ]


    confidence_percentage = (
        confidence.item() * 100
    )


    # -----------------------------------------------------
    # Return JSON
    # -----------------------------------------------------

    return {

        "prediction":
            predicted_class,

        "confidence":
            round(
                confidence_percentage,
                2
            )
    }