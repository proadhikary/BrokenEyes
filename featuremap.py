import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Paths
models_path = 'models'
input_image_path = 'output/test.png'
output_dir = 'feature_map_differences-1'

# Disorders list
disorders = ['normal', 'amd', 'cataract', 'glaucoma', 'refractive', 'retinopathy']

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the input image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

input_image = load_image(input_image_path)

# Function to match architecture from training
def modify_resnet_for_disorders():
    model = models.resnet18(pretrained=False)  # Load ResNet without pretraining
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    
    # Define the exact fc structure from training
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2),  # Binary classification
        nn.LogSoftmax(dim=1)
    )
    return model

# Function to extract feature maps for the final convolutional layer
def extract_feature_map(model, image):
    model.eval()
    activations = []

    # Hook to capture the output of the final convolutional layer
    def hook_fn(module, input, output):
        activations.append(output)

    # Register hook on the last layer of ResNet18 (layer4)
    hook = model.layer4[-1].register_forward_hook(hook_fn)

    # Forward pass
    _ = model(image)
    hook.remove()  # Remove the hook after forward pass

    return activations[0]  # Return the captured activation

# Compute and visualize differences between feature maps
def compare_feature_maps(feature_map_normal, feature_map_disorder, disorder_name):
    # Calculate the absolute difference
    diff_map = torch.abs(feature_map_normal - feature_map_disorder)
    
    # Average over all channels
    avg_diff_map = diff_map.mean(dim=1).squeeze().detach().cpu().numpy()
    
    # Plot heatmap of the differences
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_diff_map, cmap='hot')
    plt.colorbar()
    plt.title(f'Feature Map Difference: Normal vs {disorder_name.capitalize()}')
    
    # Save the heatmap
    output_path = os.path.join(output_dir, f"feature_map_diff_{disorder_name}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Difference heatmap saved for {disorder_name}: {output_path}")
    
    # Return quantitative metrics
    activation_energy_normal = torch.sum(torch.abs(feature_map_normal)).item()
    activation_energy_disorder = torch.sum(torch.abs(feature_map_disorder)).item()
    similarity = torch.nn.functional.cosine_similarity(
        feature_map_normal.flatten(), feature_map_disorder.flatten(), dim=0
    ).item()
    return activation_energy_normal, activation_energy_disorder, similarity

# Main processing loop
feature_maps = {}
metrics = []

# Load the normal model and compute its feature map
normal_model_path = os.path.join(models_path, 'normal.pth')
normal_model = modify_resnet_for_disorders()
normal_model.load_state_dict(torch.load(normal_model_path))
feature_map_normal = extract_feature_map(normal_model, input_image)

# Process each disorder
for disorder in disorders:
    if disorder == 'normal':
        continue  # Skip normal since it's already processed

    model_path = os.path.join(models_path, f'{disorder}.pth')
    if not os.path.exists(model_path):
        print(f"Model for {disorder} not found. Skipping...")
        continue

    # Load the model for the current disorder
    model = modify_resnet_for_disorders()
    model.load_state_dict(torch.load(model_path))

    # Extract the feature map
    feature_map_disorder = extract_feature_map(model, input_image)

    # Compare and save heatmap
    activation_energy_normal, activation_energy_disorder, similarity = compare_feature_maps(
        feature_map_normal, feature_map_disorder, disorder
    )
    
    # Save metrics
    metrics.append({
        'Disorder': disorder,
        'Normal Activation Energy': activation_energy_normal,
        'Disorder Activation Energy': activation_energy_disorder,
        'Cosine Similarity': similarity
    })

# Display metrics
import pandas as pd
metrics_df = pd.DataFrame(metrics)
metrics_csv_path = os.path.join(output_dir, 'metrics_comparison.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved to: {metrics_csv_path}")