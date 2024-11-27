![BrokenEyes Simulation](https://raw.githubusercontent.com/proadhikary/BrokenEyes/refs/heads/main/results/processed-input/elon.png)

**BrokenEyes** is a computational framework designed to simulate the effects of five common eye disorders: **Age-related Macular Degeneration (AMD)**, **Cataract**, **Glaucoma**, **Refractive Errors**, and **Diabetic Retinopathy**. The system generates realistic visual impairments and analyzes their impact on neural-like feature representations in deep learning models.

## Features
- Simulation of vision disorders using the custom **BrokenEyes** filter system.
- Training and evaluation of deep learning models on human and non-human datasets under normal and impaired conditions.
- Quantitative analysis using metrics such as **Activation Energy** and **Cosine Similarity**.
- Visual comparison of feature maps to identify disorder-specific disruptions.

## Dataset
The framework uses a combination of:
- **Labelled Faces in the Wild (LFW)** dataset for "human" images.
- **MS-COCO 2017** dataset for "non-human" images.
Both datasets are augmented with disorder-specific filters to create realistic visual impairments.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/BrokenEyes.git
```
```bash
cd BrokenEyes
```
```bash
pip install -r requirements.txt
```

## Usage
- Train the normal and disorder-specific models.
- Use the feature extraction module to compare feature maps.
- Visualize and analyze the disruptions caused by different disorders.

## Results
- Cataract and Glaucoma showed the most significant disruptions in feature maps.
- Evaluation metrics quantified the alignment between normal and disorder-specific feature representations.


## Acknowledgement
This is a term project for ELL890 - Computational Neuroscience.
