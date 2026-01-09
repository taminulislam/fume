"""
FUME Training Script
Trains FUME-FastSCNN model for dual-gas acidosis detection
"""

import os
import torch
import yaml
from pathlib import Path
from models import FUMEFastSCNN
from train import Trainer


def main():
    # Configuration
    config_path = 'configs/fume_fastscnn_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Model: {config['model']['name']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['optimizer']['lr']}")

    # Check data files
    required_files = [
        'data/paired_train_annotations.csv',
        'data/paired_val_annotations.csv',
        'data/paired_test_annotations.csv'
    ]

    if not all(os.path.exists(f) for f in required_files):
        print("Error: Paired annotation files not found. Run data/pairing.py first.")
        return

    # Train
    print("\nStarting training...")
    trainer = Trainer(config_path=config_path)
    trainer.train()

    print("\nTraining completed!")
    print("Checkpoints saved in: checkpoints/")


if __name__ == "__main__":
    main()
