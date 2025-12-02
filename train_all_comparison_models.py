"""
Batch Training Script for All Comparison Models
Trains all 8 comparison models + FUME-FastSCNN for benchmarking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import subprocess
import sys

# Import all models
from models import (
    FUMEFastSCNN,
    BiSeNetV2,
    CMX,
    DDRNetSlim,
    RTFNet,
    ESPNetV2,
    MTINet,
    ENet,
    DANet
)

from data import FUMEDataset, get_train_transforms, get_val_transforms
from losses import MultiTaskLoss
from utils.metrics import SegmentationMetrics, ClassificationMetrics
from utils.logger import get_logger
from utils.visualization import visualize_dual_gas_results


# Model registry
MODEL_REGISTRY = {
    'FUMEFastSCNN': FUMEFastSCNN,
    'BiSeNetV2': BiSeNetV2,
    'CMX': CMX,
    'DDRNetSlim': DDRNetSlim,
    'RTFNet': RTFNet,
    'ESPNetV2': ESPNetV2,
    'MTINet': MTINet,
    'ENet': ENet,
    'DANet': DANet
}


class FastTrainer:
    """Fast trainer for comparison models"""

    def __init__(self, config_path: str, model_name: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = model_name

        # Update experiment name
        self.config['experiment']['name'] = f"{model_name}-Fast"
        self.config['model']['name'] = model_name

        # Set seed
        self.set_seed(self.config['experiment']['seed'])

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*70}")
        print(f"🚀 Training {model_name}")
        print(f"{'='*70}")
        print(f"Device: {self.device}")

        # Create directories
        self.setup_directories()

        # Initialize logger
        self.setup_logger()

        # Load data
        self.setup_data()

        # Build model
        self.setup_model()

        # Setup loss
        self.setup_loss()

        # Setup optimizer and scheduler
        self.setup_optimizer()

        # Metrics
        self.setup_metrics()

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_time = 0.0

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def setup_directories(self):
        for key, path in self.config['directories'].items():
            model_dir = Path(path) / self.model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            self.config['directories'][key] = str(model_dir)

    def setup_logger(self):
        exp_config = self.config['experiment']

        if exp_config['use_wandb']:
            self.logger = get_logger(
                use_wandb=True,
                project_name=exp_config['project'],
                experiment_name=exp_config['name'],
                config=self.config,
                entity=exp_config.get('wandb_entity'),
                tags=exp_config.get('tags', [])
            )
        else:
            self.logger = get_logger(
                use_wandb=False,
                log_dir=self.config['directories']['logs']
            )

    def setup_data(self):
        data_config = self.config['data']

        # Transforms
        train_transform = get_train_transforms(tuple(data_config['image_size']))
        val_transform = get_val_transforms(tuple(data_config['image_size']))

        # Datasets
        self.train_dataset = FUMEDataset(
            paired_csv=data_config['paired_train_csv'],
            dataset_root=data_config['dataset_root'],
            transform=train_transform,
            modality_dropout=data_config['modality_dropout'],
            is_training=True
        )

        self.val_dataset = FUMEDataset(
            paired_csv=data_config['paired_val_csv'],
            dataset_root=data_config['dataset_root'],
            transform=val_transform,
            modality_dropout=0.0,
            is_training=False
        )

        # Weighted sampling
        if self.config.get('weighted_sampling', True):
            sample_weights = self.train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        # Dataloaders
        train_config = self.config['training']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            sampler=sampler,
            shuffle=shuffle,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory']
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory']
        )

        print(f"✅ Data: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

    def setup_model(self):
        model_config = self.config['model']

        # Get model class
        ModelClass = MODEL_REGISTRY[self.model_name]

        # Instantiate model
        self.model = ModelClass(
            num_classes=model_config['num_classes'],
            num_seg_classes=model_config['num_seg_classes']
        ).to(self.device)

        # Log model info
        num_params = self.model.get_num_parameters()
        print(f"✅ Model: {num_params:,} parameters ({num_params/1e6:.2f}M)")

    def setup_loss(self):
        loss_config = self.config['training']['loss']

        cls_alpha = torch.tensor(loss_config['cls_alpha']).to(self.device)

        self.criterion = MultiTaskLoss(
            seg_loss_weight=loss_config['seg_weight'],
            cls_loss_weight=loss_config['cls_weight'],
            cls_alpha=cls_alpha,
            cls_gamma=loss_config['focal_gamma'],
            seg_gamma=loss_config['focal_gamma'],
            use_focal_dice=loss_config['use_focal_dice']
        ).to(self.device)

    def setup_optimizer(self):
        opt_config = self.config['training']['optimizer']
        sch_config = self.config['training']['scheduler']

        if opt_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_config['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )

        if sch_config['type'] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sch_config['T_max'],
                eta_min=sch_config['eta_min']
            )
        elif sch_config['type'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None

        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['use_amp'] else None

    def setup_metrics(self):
        self.seg_metrics = SegmentationMetrics(num_classes=self.config['model']['num_seg_classes'])
        self.cls_metrics = ClassificationMetrics(
            num_classes=self.config['model']['num_classes'],
            class_names=self.config['data']['class_names']
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")

        for batch in pbar:
            co2_frame = batch['co2_frame'].to(self.device)
            ch4_frame = batch['ch4_frame'].to(self.device)
            co2_mask = batch['co2_mask'].to(self.device)
            ch4_mask = batch['ch4_mask'].to(self.device)
            class_label = batch['class_label'].to(self.device)
            modality_mask = batch['modality_mask'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.config['training']['use_amp']):
                outputs = self.model(co2_frame, ch4_frame, modality_mask)

                targets = {
                    'class_label': class_label,
                    'co2_mask': co2_mask,
                    'ch4_mask': ch4_mask,
                    'modality_mask': modality_mask
                }

                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']

            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.config['training']['grad_clip']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config['training']['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                self.optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return np.mean(epoch_losses)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_losses = []

        self.seg_metrics.reset()
        self.cls_metrics.reset()

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            co2_frame = batch['co2_frame'].to(self.device)
            ch4_frame = batch['ch4_frame'].to(self.device)
            co2_mask = batch['co2_mask'].to(self.device)
            ch4_mask = batch['ch4_mask'].to(self.device)
            class_label = batch['class_label'].to(self.device)
            modality_mask = batch['modality_mask'].to(self.device)

            outputs = self.model(co2_frame, ch4_frame, modality_mask)

            targets = {
                'class_label': class_label,
                'co2_mask': co2_mask,
                'ch4_mask': ch4_mask,
                'modality_mask': modality_mask
            }

            loss_dict = self.criterion(outputs, targets)
            val_losses.append(loss_dict['total_loss'].item())

            pred_cls = outputs['cls_logits'].argmax(dim=1)
            self.cls_metrics.update(pred_cls, class_label)

            pred_seg = outputs['co2_seg_logits'].argmax(dim=1)
            self.seg_metrics.update(pred_seg, co2_mask)

        seg_results = self.seg_metrics.compute()
        cls_results = self.cls_metrics.compute()

        return np.mean(val_losses), seg_results, cls_results

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'training_time': self.training_time,
            'config': self.config
        }

        checkpoint_dir = Path(self.config['directories']['checkpoints'])

        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

        last_path = checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)

    def train(self):
        print(f"\n🚀 Starting Fast Training ({self.config['training']['num_epochs']} epochs)")

        start_time = time.time()

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, seg_results, cls_results = self.validate()

            # Check if best
            val_metric = cls_results['balanced_accuracy']
            is_best = val_metric > self.best_metric
            if is_best:
                self.best_metric = val_metric

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Print summary
            print(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Balanced Acc: {cls_results['balanced_accuracy']:.4f} | mIoU: {seg_results['mean_iou']:.4f}")
            print(f"  Best Balanced Acc: {self.best_metric:.4f}")
            print(f"  Time: {epoch_time/60:.2f} min")
            print("-" * 70)

        self.training_time = time.time() - start_time

        print(f"\n✅ Training completed in {self.training_time/60:.2f} minutes!")
        print(f"📊 Best Balanced Accuracy: {self.best_metric:.4f}")

        self.logger.finish()

        return {
            'model_name': self.model_name,
            'best_metric': self.best_metric,
            'training_time': self.training_time,
            'final_results': {
                'seg': seg_results,
                'cls': cls_results
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Train all comparison models")
    parser.add_argument('--config', type=str, default='configs/fast_comparison_config.yaml')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['FUMEFastSCNN', 'BiSeNetV2', 'CMX', 'DDRNetSlim',
                               'RTFNet', 'ESPNetV2', 'MTINet', 'ENet', 'DANet'],
                       help='Models to train')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 BATCH TRAINING: COMPARISON MODELS")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Models: {', '.join(args.models)}")
    print("="*70 + "\n")

    results = []

    for model_name in args.models:
        try:
            trainer = FastTrainer(args.config, model_name)
            result = trainer.train()
            results.append(result)

            # Clear CUDA cache
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)

    results_sorted = sorted(results, key=lambda x: x['best_metric'], reverse=True)

    for i, result in enumerate(results_sorted, 1):
        print(f"\n{i}. {result['model_name']}")
        print(f"   Best Balanced Acc: {result['best_metric']:.4f}")
        print(f"   Training Time: {result['training_time']/60:.2f} min")
        print(f"   mIoU: {result['final_results']['seg']['mean_iou']:.4f}")
        print(f"   F1 (macro): {result['final_results']['cls']['macro_f1']:.4f}")

    print("\n" + "="*70)
    print("✅ All models trained successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
