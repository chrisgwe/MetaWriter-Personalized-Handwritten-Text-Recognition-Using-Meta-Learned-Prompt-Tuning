# MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning

![MetaWriter Architecture](https://github.com/user-attachments/assets/0baa1f6d-6828-4ecf-86c0-4895b8ec9361)

## Overview
MetaWriter introduces a novel handwritten text recognition approach that combines meta-learning with prompt tuning to adapt to individual handwriting styles. This method enables personalized recognition without extensive retraining.

## Requirements
Essential software dependencies to run MetaWriter:
- **Python**: 3.9 (Required for compatibility)
- **Cuda**: 11.7 
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
{
    "transfer_learning": {
        "encoder": [],
        "decoder": []
    }
}
```
*Note: Add paths to pretrained encoder/decoder weights inside the brackets*

### 3. Run Training

Execute the main script example:

```bash
python main.py \
  --dataset_name [your_dataset] \        # Specify dataset (IAM/RIMES)
  --dataset_level [dataset_level] \      # Choose processing level
  --dataset_variant _sem \               # Dataset variant suffix
  --batch_size 1 \                       # Samples per batch
  --max_epochs 5000 \                    # Maximum training epochs
  --learning_rate 0.0001 \               # Initial learning rate
  --transfer_learning_weights [path_to_weights_folder] \  # Pretrained weights
  --output_dir [output_folder] \         # Results directory
  --use_amp                              # Enable automatic mixed precision
```

## Citation


```bibtex
@inproceedings{gu2025metawriter,
  title={MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning},
  author={Gu, Wenhao and Gu, Li and Suen, Ching Yee and Wang, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
