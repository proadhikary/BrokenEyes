import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Specify paths
models_path = 'inputs'
input_dirs = {
    "normal": "output/test.png",
    "amd": "output/amd.png",
    "cataract": "output/cataract.png",
    "glaucoma": "output/glucoma.png",
    "refractive": "output/refractive.png",
    "retinopathy": "output/retinopathy.png",
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 2),
    nn.LogSoftmax(dim=1)
)
model_path = 'models/normal.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Image transformation (adjust as per model requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to model input size
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define class names
class_names = ['humans', 'non-humans']

# Define a function to predict the class of a single image
def predict_image(image_path, model, transform, class_names):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict the class
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.exp(output)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return class_names[predicted_class], confidence
    except Exception as e:
        return f"Error processing {image_path}: {e}", 0.0

# Output file
output_file = 'results.txt'

# Perform inference and save results
with open(output_file, 'w') as f:
    for label, image_path in input_dirs.items():
        print(f"Processing {label}...")
        image_full_path = os.path.join(image_path)  # Construct full path
        predicted_class, confidence = predict_image(image_full_path, model, transform, class_names)
        f.write(f"{label}:\nPredicted Class: {predicted_class}\nConfidence: {confidence:.4f}\n\n")

print(f"Results saved to {output_file}")
