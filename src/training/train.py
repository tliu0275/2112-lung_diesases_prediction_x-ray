"""
Complete training pipeline for pneumonia detection model

Features:
- Multi-task learning
- Mixed precision training
- Early stopping
- Checkpoint saving
- TensorBoard logging
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import yaml
import argparse
import logging
from typing import Dict, Tuple, Union
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.medical_models import create_model
from src.preprocessing.dicom_xray_loader import create_data_loaders
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _metrics_for_yaml(metrics: Dict) -> Dict:
    """Convert numpy scalars/arrays so yaml.safe_dump does not fail."""

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    return convert(metrics)


class PneumoniaTrainer:
    """Trainer class for pneumonia detection model"""
    
    def __init__(self, config_path_or_dict: Union[str, Path, Dict], device: str = 'cuda'):
        """Initialize trainer from YAML path or an in-memory config dict."""
        if isinstance(config_path_or_dict, dict):
            self.config = dict(config_path_or_dict)
        else:
            with open(config_path_or_dict, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        self.device = device
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.model_type = self.config['model'].get('model_type', 'multi_task')
        
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        self.model = create_model(
            model_type=self.config['model']['model_type'],
            backbone=self.config['model']['name'],
            pretrained=self.config['model']['pretrained'],
            device=device,
            num_classes=self.config['model'].get('num_classes'),
            severity_classes=self.config['model'].get('severity_classes'),
            tabular_dim=int(self.config['model'].get('tabular_dim', 0)),
        )
        
        # Create optimizer
        tcfg = self.config["training"]
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(tcfg["learning_rate"]),
            weight_decay=float(tcfg["weight_decay"]),
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )
        
        # Gradient scaler (enabled only when mixed precision is requested and CUDA is available)
        mp_requested = bool(self.config['training'].get('mixed_precision', True))
        self.use_mixed_precision = mp_requested and torch.cuda.is_available() and device.startswith('cuda')
        self.scaler = GradScaler(enabled=self.use_mixed_precision)
        
        # Metrics calculator (binary or multiclass)
        self.metrics = MetricsCalculator(task='classification')
        
        logger.info(f"Trainer initialized on {device}")

    def _forward(self, batch):
        images = batch["image"].to(self.device)
        if self.model_type == "binary":
            return self.model(images)
        tab = batch.get("tabular")
        if tab is not None:
            tab = tab.to(self.device, dtype=images.dtype)
        return self.model(images, tab)

    def _compute_loss(self, outputs, batch):
        """Compute loss for binary or multi-task model."""
        labels = batch['label'].to(self.device)

        if self.model_type == "binary":
            logits = outputs
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, {"loss": loss.detach()}

        # multi_task
        binary_logits, severity_logits = outputs
        binary_labels = labels
        if 'severity' not in batch:
            raise ValueError("Multi-task training requires 'severity' labels in the dataset/batch.")
        severity_labels = batch['severity'].to(self.device)

        binary_loss = nn.CrossEntropyLoss()(binary_logits, binary_labels)
        severity_loss = nn.CrossEntropyLoss()(severity_logits, severity_labels)

        lw = self.config['training']['loss_weights']
        binary_weight = lw.get('classification', lw.get('binary_classification', 0.6))
        severity_weight = lw['severity_prediction']

        total_loss = binary_weight * binary_loss + severity_weight * severity_loss
        return total_loss, {
            "loss": total_loss.detach(),
            "binary_loss": binary_loss.detach(),
            "severity_loss": severity_loss.detach(),
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            with autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu', enabled=self.use_mixed_precision):
                outputs = self._forward(batch)
                loss, _ = self._compute_loss(outputs, batch)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config['training']['gradient_clip_max_norm'])
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{total_loss / (progress_bar.n + 1):.4f}"})
        
        return {'loss': total_loss / len(train_loader)}
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_pos_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                with autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu', enabled=self.use_mixed_precision):
                    outputs = self._forward(batch)
                    loss, _ = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()

                labels = batch['label'].cpu().numpy()

                if self.model_type == 'binary':
                    logits = outputs
                else:
                    logits, _ = outputs  # use binary head for reporting

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                if probs.shape[1] == 2:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs[np.arange(len(preds)), preds]

                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())
                all_pos_probs.extend(pos_probs.tolist())

        self.metrics.reset()
        self.metrics.add_batch(all_preds, all_pos_probs, all_targets)
        metrics = self.metrics.calculate_metrics()
        metrics['loss'] = total_loss / len(val_loader)

        # keep compatibility with old logging key name used elsewhere
        metrics['binary_f1'] = metrics.get('f1', 0.0)
        return metrics

    def evaluate(self, loader: DataLoader, name: str = "test") -> Dict:
        """Evaluate a loader and return metrics dict."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_pos_probs = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({name})"):
                labels = batch['label'].cpu().numpy()

                with autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu', enabled=self.use_mixed_precision):
                    outputs = self._forward(batch)

                if self.model_type == 'binary':
                    logits = outputs
                else:
                    logits, _ = outputs

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                if probs.shape[1] == 2:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs[np.arange(len(preds)), preds]

                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())
                all_pos_probs.extend(pos_probs.tolist())

        self.metrics.reset()
        self.metrics.add_batch(all_preds, all_pos_probs, all_targets)
        metrics = self.metrics.calculate_metrics()
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0):
        """Full training loop. start_epoch: resume index (0-based next epoch to run)."""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']

        if start_epoch >= num_epochs:
            logger.info(
                "start_epoch (%s) >= num_epochs (%s); skipping training loop.",
                start_epoch,
                num_epochs,
            )
            return

        if start_epoch > 0:
            logger.info("Resuming: epochs %s..%s", start_epoch + 1, num_epochs)
        else:
            logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(start_epoch, num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            if (epoch + 1) % self.config['evaluation']['eval_freq'] == 0:
                val_metrics = self.validate(val_loader, epoch)
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['binary_f1']:.4f}")
                
                if val_metrics['binary_f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['binary_f1']
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                    logger.info(f"✅ Best model saved! F1: {self.best_val_f1:.4f}")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            self.scheduler.step()
        
        logger.info("✅ Training completed!")

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Load model/optimizer/scheduler (if present); return next epoch index to train."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_f1 = float(ckpt.get("best_val_f1", 0.0))

        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            saved = int(ckpt.get("epoch", -1))
            for _ in range(saved + 1):
                self.scheduler.step()

        if self.use_mixed_precision and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        start = int(ckpt["epoch"]) + 1
        logger.info("Resumed from %s; continuing from epoch %s", path, start + 1)
        return start

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1
        }
        if self.use_mixed_precision:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/configs/config.yaml')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-dir', type=str, default=None, help='Override data.data_dir (e.g. data_folder_2)')
    parser.add_argument('--epochs', type=int, default=None, help='Override training.num_epochs')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Override model.num_classes (must match number of class folders)',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to .pth checkpoint (loads model/optimizer/scheduler; continues toward num_epochs)',
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        repo = Path(__file__).resolve().parents[2]
        alt = repo / args.config
        if alt.is_file():
            cfg_path = alt
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.data_dir is not None:
        cfg.setdefault('data', {})['data_dir'] = args.data_dir
    if args.epochs is not None:
        cfg.setdefault('training', {})['num_epochs'] = int(args.epochs)
        ne = int(cfg['training']['num_epochs'])
        p0 = int(cfg['training'].get('early_stopping_patience', 10))
        cfg['training']['early_stopping_patience'] = max(p0, ne)
    if args.num_classes is not None:
        cfg.setdefault('model', {})['num_classes'] = int(args.num_classes)

    trainer = PneumoniaTrainer(cfg, device=args.device)

    dcfg = trainer.config["data"]
    repo_root = Path(__file__).resolve().parents[2]
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=dcfg["data_dir"],
        csv_file=dcfg.get("csv_file"),
        batch_size=trainer.config["training"]["batch_size"],
        image_size=dcfg.get("image_size", 224),
        train_split=dcfg.get("train_split", 0.7),
        val_split=dcfg.get("val_split", 0.15),
        test_split=dcfg.get("test_split", 0.15),
        num_workers=dcfg["num_workers"],
        augment_train=dcfg["augment_train"],
        seed=dcfg["seed"],
        severity_strategy=dcfg.get("severity_strategy", "auto"),
        synthetic_severity_by_class=dcfg.get("synthetic_severity_by_class"),
        severity_classes=int(trainer.config["model"].get("severity_classes", 5)),
        metadata_csv=dcfg.get("metadata_csv"),
        tabular_dim=int(trainer.config["model"].get("tabular_dim", 0)),
        tabular_extra_columns=dcfg.get("tabular_extra_columns"),
        project_root=repo_root,
    )

    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_file():
            alt = repo_root / args.resume
            if alt.is_file():
                ckpt_path = alt
            else:
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        start_epoch = trainer.load_checkpoint(ckpt_path)

    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    # Optional: evaluate on test split if present
    if test_loader is not None:
        # Load best checkpoint for evaluation if available
        best_path = trainer.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(ckpt['model_state_dict'])
        test_metrics = trainer.evaluate(test_loader, name="test")
        logger.info(f"Test - Acc: {test_metrics.get('accuracy', 0.0):.4f}, F1: {test_metrics.get('f1', 0.0):.4f}")

        # Save metrics to outputs/
        out_dir = Path(trainer.config['paths'].get('output_dir', 'outputs/'))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'test_metrics.yaml', 'w') as f:
            yaml.safe_dump(_metrics_for_yaml(test_metrics), f, sort_keys=False)


if __name__ == '__main__':
    main()
