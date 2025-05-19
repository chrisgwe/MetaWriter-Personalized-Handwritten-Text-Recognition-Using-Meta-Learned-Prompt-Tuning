# MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning

![MetaWriter Architecture](https://github.com/user-attachments/assets/0baa1f6d-6828-4ecf-86c0-4895b8ec9361)

## Overview
MetaWriter introduces a novel handwritten text recognition approach that combines meta-learning with prompt tuning to adapt to individual handwriting styles. This method enables personalized recognition without extensive retraining.

## Requirements
Essential software dependencies to run MetaWriter:
- **Python**: 3.9 (Required for compatibility)
- **PyTorch**: 2.0.1 (Deep learning framework)

## Usage

### 1. Dataset Preparation
Prepare your training data by downloading and organizing these standard handwritten text datasets:

- **RIMES**: [Download from Teklia](https://teklia.com/research/rimes-database/)  
  French handwritten text dataset with various writing styles
- **IAM**: [Download from FKI](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)  
  English handwritten text database containing forms and lines

Store the datasets in the `data/` directory or specify an alternative path in the configuration.

### 2. Transfer Learning Weights Configuration
Initialize the model with synthetic data weights by modifying the configuration:

```json
"transfer_learning": {
    "encoder": [],  // Path to pretrained encoder weights
    "decoder": []   // Path to pretrained decoder weights
}
```
### 3. Run Training
Execute the main script example:
```json
python main.py \
  --dataset_name [your dataset] \        # Specify dataset (IAM/RIMES)
  --dataset_level [dataset level] \      # Choose processing level
  --dataset_variant _sem \               # Dataset variant suffix
  --batch_size 1 \                       # Samples per batch
  --max_epochs 5000 \                    # Maximum training epochs
  --learning_rate 0.0001 \               # Initial learning rate
  --transfer_learning_weights [location to your folder] \  # Pretrained weights
  --output_dir [location to your folder] \  # Results directory
  --use_amp                             # Enable automatic mixed precision
```
