"""
FUME Training Script
Train FUME-FastSCNN and baseline models
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

# Import FUME modules
from models import FUMEFastSCNN
from data import FUMEDataset, get_train_transforms, get_val_transforms
from losses import MultiTaskLoss
from utils.metrics import SegmentationMetrics, ClassificationMetrics
from utils.logger import get_logger
from utils.visualization import visualize_dual_gas_results


class Trainer:
    """FUME Trainer with W&B logging and checkpointing"""

    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set seed
        self.set_seed(self.config['experiment']['seed'])

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")

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
        self.patience_counter = 0

    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print(f"✅ Seed set to {seed}")

    def setup_directories(self):
        """Create necessary directories"""
        for key, path in self.config['directories'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
        print("✅ Directories created")

    def setup_logger(self):
        """Initialize W&B logger"""
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

        print("✅ Logger initialized")

    def setup_data(self):
        """Setup dataloaders"""
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

        # Weighted sampling for class imbalance
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

        print(f"✅ Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

    def setup_model(self):
        """Initialize model"""
        model_config = self.config['model']

        self.model = FUMEFastSCNN(
            num_classes=model_config['num_classes'],
            num_seg_classes=model_config['num_seg_classes'],
            shared_encoder=model_config['shared_encoder']
        ).to(self.device)

        # Log model info
        num_params = self.model.get_num_parameters()
        print(f"✅ Model created: {num_params:,} parameters ({num_params/1e6:.2f}M)")

        if hasattr(self.logger, 'log_model'):
            self.logger.log_model(self.model)

    def setup_loss(self):
        """Initialize loss function"""
        loss_config = self.config['training']['loss']

        # Class weights
        cls_alpha = torch.tensor(loss_config['cls_alpha']).to(self.device)

        self.criterion = MultiTaskLoss(
            seg_loss_weight=loss_config['seg_weight'],
            cls_loss_weight=loss_config['cls_weight'],
            cls_alpha=cls_alpha,
            cls_gamma=loss_config['focal_gamma'],
            seg_gamma=loss_config['focal_gamma'],
            use_focal_dice=loss_config['use_focal_dice']
        ).to(self.device)

        print("✅ Loss function initialized")

    def setup_optimizer(self):
        """Initialize optimizer and scheduler"""
        opt_config = self.config['training']['optimizer']
        sch_config = self.config['training']['scheduler']

        # Optimizer
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

        # Scheduler
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

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['use_amp'] else None

        print("✅ Optimizer and scheduler initialized")

    def setup_metrics(self):
        """Initialize metrics"""
        self.seg_metrics = SegmentationMetrics(num_classes=self.config['model']['num_seg_classes'])
        self.cls_metrics = ClassificationMetrics(
            num_classes=self.config['model']['num_classes'],
            class_names=self.config['data']['class_names']
        )
        print("✅ Metrics initialized")

    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            co2_frame = batch['co2_frame'].to(self.device)
            ch4_frame = batch['ch4_frame'].to(self.device)
            co2_mask = batch['co2_mask'].to(self.device)
            ch4_mask = batch['ch4_mask'].to(self.device)
            class_label = batch['class_label'].to(self.device)
            modality_mask = batch['modality_mask'].to(self.device)

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.config['training']['use_amp']):
                outputs = self.model(co2_frame, ch4_frame, modality_mask)

                # Compute loss
                targets = {
                    'class_label': class_label,
                    'co2_mask': co2_mask,
                    'ch4_mask': ch4_mask,
                    'modality_mask': modality_mask
                }

                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']

            # Backward pass
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

            # Track loss
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

            # Log metrics
            if batch_idx % self.config['logging']['log_freq'] == 0:
                self.logger.log_metrics({
                    'train_loss': loss.item(),
                    'train_cls_loss': loss_dict['cls_loss'].item(),
                    'train_seg_loss': loss_dict['seg_loss'].item()
                }, step=epoch * len(self.train_loader) + batch_idx, prefix='train/')

        return np.mean(epoch_losses)

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()
        val_losses = []

        # Reset metrics
        self.seg_metrics.reset()
        self.cls_metrics.reset()

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            co2_frame = batch['co2_frame'].to(self.device)
            ch4_frame = batch['ch4_frame'].to(self.device)
            co2_mask = batch['co2_mask'].to(self.device)
            ch4_mask = batch['ch4_mask'].to(self.device)
            class_label = batch['class_label'].to(self.device)
            modality_mask = batch['modality_mask'].to(self.device)

            # Forward pass
            outputs = self.model(co2_frame, ch4_frame, modality_mask)

            # Compute loss
            targets = {
                'class_label': class_label,
                'co2_mask': co2_mask,
                'ch4_mask': ch4_mask,
                'modality_mask': modality_mask
            }

            loss_dict = self.criterion(outputs, targets)
            val_losses.append(loss_dict['total_loss'].item())

            # Update metrics
            pred_cls = outputs['cls_logits'].argmax(dim=1)
            self.cls_metrics.update(pred_cls, class_label)

            # Segmentation metrics (CO2 only for simplicity)
            pred_seg = outputs['co2_seg_logits'].argmax(dim=1)
            self.seg_metrics.update(pred_seg, co2_mask)

        # Compute metrics
        seg_results = self.seg_metrics.compute()
        cls_results = self.cls_metrics.compute()

        # Log metrics
        val_loss = np.mean(val_losses)
        self.logger.log_metrics({
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in seg_results.items() if isinstance(v, (int, float))},
            **{f'val_{k}': v for k, v in cls_results.items() if isinstance(v, (int, float))}
        }, step=epoch)

        return cls_results['balanced_accuracy'], val_loss, seg_results, cls_results

    def print_metrics_table(self, epoch: int, train_loss: float, val_loss: float,
                            seg_results: dict, cls_results: dict):
        """Print a formatted table of evaluation metrics"""
        print("\n" + "=" * 70)
        print(f"  EPOCH {epoch} RESULTS")
        print("=" * 70)

        print("\n  LOSS")
        print("-" * 40)
        print(f"  {'Train Loss':<25} {train_loss:.4f}")
        print(f"  {'Val Loss':<25} {val_loss:.4f}")

        print("\n  SEGMENTATION METRICS")
        print("-" * 40)
        seg_keys = ['mean_iou', 'mean_dice', 'pixel_accuracy']
        for key in seg_keys:
            if key in seg_results:
                print(f"  {key:<25} {seg_results[key]:.4f}")

        print("\n  CLASSIFICATION METRICS")
        print("-" * 40)
        cls_keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'cohens_kappa']
        for key in cls_keys:
            if key in cls_results:
                print(f"  {key:<25} {cls_results[key]:.4f}")

        print("\n  PER-CLASS F1 SCORES")
        print("-" * 40)
        class_names = self.config['data']['class_names']
        for name in class_names:
            key = f'{name}_f1'
            if key in cls_results:
                print(f"  {name:<25} {cls_results[key]:.4f}")

        print("=" * 70 + "\n")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        checkpoint_dir = Path(self.config['directories']['checkpoints'])
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved: {best_path}")

        # Save last
        last_path = checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)

    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 Starting Training")
        print("="*70)

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            if epoch % self.config['validation']['val_freq'] == 0:
                val_metric, val_loss, seg_results, cls_results = self.validate(epoch)

                # Check if best
                is_best = val_metric > self.best_metric
                if is_best:
                    self.best_metric = val_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint
                if epoch % self.config['training']['checkpoint']['save_freq'] == 0:
                    self.save_checkpoint(epoch, is_best)

                # Print metrics table every 5 epochs
                if epoch % 5 == 0:
                    self.print_metrics_table(epoch, train_loss, val_loss, seg_results, cls_results)

                # Early stopping
                early_stop_config = self.config['training']['early_stopping']
                if self.patience_counter >= early_stop_config['patience']:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                    break

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log epoch time
            epoch_time = time.time() - epoch_start
            self.logger.log_epoch_time(epoch, epoch_time)

            print(f"Epoch {epoch} completed in {epoch_time/60:.2f} minutes")
            print(f"Best metric: {self.best_metric:.4f}")
            print("-"*70)

        print("\n✅ Training completed!")
        self.logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Train FUME-FastSCNN")
    parser.add_argument('--config', type=str, default='configs/fume_fastscnn_config.yaml',
                      help='Path to config file')
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
