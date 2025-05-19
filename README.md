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

### 2. transfer_learning Weights
Load the synthetic data weights before training:
"transfer_learning": {
                "encoder": [],
                "decoder": [],
            },

### 3. Run Training
Execute the main script example:
python main.py \
  --dataset_name [your dataset] \
  --dataset_level [dataset level] \
  --dataset_variant _sem \
  --batch_size 1 \
  --max_epochs 5000 \
  --learning_rate 0.0001 \
  --transfer_learning_weights [location to your folder] \
  --output_dir [location to your folder] \
  --use_amp 
