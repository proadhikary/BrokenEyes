import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import accuracy_score
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Disorder-specific datasets
data_dirs = {
    "normal": "data/normal-data",
    "amd": "data/amd-data",
    "cataract": "data/cataract-data",
    "glaucoma": "data/glaucoma-data",
    "refractive": "data/refractive-data",
    "retinopathy": "data/retinopathy-data",
}

# Create a directory for saving models
os.makedirs("models", exist_ok=True)

# Initialize a results file
results_file = "results.txt"
with open(results_file, "w") as f:
    f.write("Disorder\tTest Accuracy\n")

# Function to train and evaluate a model for a specific dataset
def train_and_evaluate(data_dir, disorder_name):
    # Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split the dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the model (Transfer Learning with ResNet18)
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer for binary classification
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2),
        nn.LogSoftmax(dim=1)
    )
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training loop
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5):
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            valid_loss = 0
            valid_preds, valid_targets = [], []

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    valid_preds.extend(preds.cpu().numpy())
                    valid_targets.extend(labels.cpu().numpy())

            valid_acc = accuracy_score(valid_targets, valid_preds)
            print(f"Epoch {epoch+1}/5, "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Valid Loss: {valid_loss/len(valid_loader):.4f}, "
                  f"Valid Accuracy: {valid_acc:.4f}")

    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5)

    # Save the trained model
    model_path = f"models/{disorder_name}.pth"
    torch.save(model.state_dict(), model_path)

    # Evaluate on test data
    model.eval()
    test_preds, test_targets = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    print(f"Test Accuracy for {disorder_name}: {test_acc:.4f}")

    # Log the results to the results file
    with open(results_file, "a") as f:
        f.write(f"{disorder_name}\t{test_acc:.4f}\n")

# Train and evaluate models for each disorder
for disorder_name, data_dir in data_dirs.items():
    print(f"Processing {disorder_name} data...")
    train_and_evaluate(data_dir, disorder_name)

print("Training and evaluation completed. Results saved to results.txt.")
