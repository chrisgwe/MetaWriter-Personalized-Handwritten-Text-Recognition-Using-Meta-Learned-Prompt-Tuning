# MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning
![image](https://github.com/user-attachments/assets/0baa1f6d-6828-4ecf-86c0-4895b8ec9361)
## Overview
MetaWriter is a novel approach to handwritten text recognition that leverages meta-learning and prompt tuning to personalize recognition for individual handwriting styles.
## Requirements
- **Python**: 3.9
- **PyTorch**: 2.0.1
## Usage

### 1. Dataset Preparation
Download the required datasets:
- **RIMES**: [Download from Teklia](https://teklia.com/research/rimes-database/)
- **IAM**: [Download from FKI](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

Place the datasets in the `data/` directory (or specify your custom path in config).

### 2. Pre-trained Weights
Load the synthetic data pre-trained weights before training:
```bash
python load_weights.py --weights path/to/synthetic_weights.pth

