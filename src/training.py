"""
Training utilities for CNN models.

This module contains training loop, validation, early stopping, and related utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import copy
from pathlib import Path


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    import time

    epoch_start = time.time()
    print(f"  [Train] Starting training epoch...")

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    batch_times = []
    data_load_times = []
    forward_times = []
    backward_times = []

    batch_start = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        data_load_time = time.time() - batch_start
        data_load_times.append(data_load_time)

        # Move to device
        transfer_start = time.time()
        images, labels = images.to(device), labels.to(device)
        transfer_time = time.time() - transfer_start

        # Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)

        # Backward pass
        backward_start = time.time()
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f"\n  [ERROR] GPU crash at batch {batch_idx}: {e}")
            print(f"  Suggestion: Reduce batch size or num_workers")
            raise
        backward_time = time.time() - backward_start
        backward_times.append(backward_time)

        # Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Log first and every 10th batch
        if batch_idx == 0:
            print(f"  [Train] Batch 0/{len(dataloader)}: {batch_time:.2f}s (data: {data_load_time:.2f}s, fwd: {forward_time:.3f}s, bwd: {backward_time:.3f}s)")
        elif (batch_idx + 1) % 10 == 0:
            avg_time = sum(batch_times[-10:]) / 10
            print(f"  [Train] Batch {batch_idx + 1}/{len(dataloader)}: avg {avg_time:.2f}s/batch, loss: {loss.item():.4f}, acc: {100 * correct / total:.2f}%")

        batch_start = time.time()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    epoch_time = time.time() - epoch_start

    # Print epoch summary
    print(f"  [Train] Epoch complete in {epoch_time:.2f}s")
    print(f"    Avg batch time: {sum(batch_times)/len(batch_times):.3f}s")
    print(f"    Avg data load: {sum(data_load_times)/len(data_load_times):.3f}s")
    print(f"    Avg forward: {sum(forward_times)/len(forward_times):.3f}s")
    print(f"    Avg backward: {sum(backward_times)/len(backward_times):.3f}s")
    print(f"    Throughput: {total/epoch_time:.1f} samples/sec")

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    import time

    val_start = time.time()
    print(f"  [Val] Starting validation...")

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log every 5th batch
            if (batch_idx + 1) % 5 == 0:
                print(f"  [Val] Batch {batch_idx + 1}/{len(dataloader)}: loss: {loss.item():.4f}, acc: {100 * correct / total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    val_time = time.time() - val_start

    print(f"  [Val] Validation complete in {val_time:.2f}s")
    print(f"    Throughput: {total/val_time:.1f} samples/sec")

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, config, model_name='model', device='cpu', save_best=True):
    """
    Complete training pipeline with early stopping and checkpointing.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration dictionary
        model_name: Name for saving the model
        device: Device to train on
        save_best: Whether to save the best model

    Returns:
        history: Dictionary containing training history
    """
    import time

    print(f"\n[Training] Moving model to device: {device}")
    transfer_start = time.time()
    model = model.to(device)
    transfer_time = time.time() - transfer_start
    print(f"[Training] Model transferred to device in {transfer_time:.3f}s")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )

    # Learning rate scheduler
    scheduler = None
    if config['training']['scheduler']['enabled']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler']['step_size'],
            gamma=config['training']['scheduler']['gamma']
        )

    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            verbose=True
        )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Training loop
    num_epochs = config['training']['epochs']
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs} | Batch size: {config['training']['batch_size']}")
    print(f"Optimizer: {config['training']['optimizer'].upper()} | LR: {config['training']['learning_rate']}")
    print(f"Early stopping: {config['training']['early_stopping']['enabled']} (patience={config['training']['early_stopping']['patience']})")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path(config['paths']['models_dir']) / f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Loading best model weights...")
                early_stopping.load_best_model(model)
                break

    print(f"\n{'='*60}")
    print(f"Training Complete: {model_name}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Epochs: {len(history['train_loss'])}")
    print(f"{'='*60}\n")

    return history


def create_dataloaders(train_hf, val_hf, test_hf, train_transform, val_transform, batch_size=16, num_workers=0, prefetch_factor=None, persistent_workers=False, cache_images=False):
    """
    Create PyTorch dataloaders from HuggingFace datasets.

    Args:
        train_hf: Training HuggingFace dataset
        val_hf: Validation HuggingFace dataset
        test_hf: Test HuggingFace dataset
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of worker processes for parallel data loading
        prefetch_factor: Number of batches to prefetch per worker (default: None/2)
        persistent_workers: Keep workers alive between epochs (reduces startup overhead)
        cache_images: Cache preprocessed images in memory (faster but uses more RAM)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from src.load_data import TrashNetDataset

    print(f"\n[DataLoader] Creating datasets and dataloaders...")
    if cache_images:
        print(f"  [DataLoader] Image caching ENABLED - will use more RAM but load faster!")

    # Create PyTorch datasets with optional caching
    train_dataset = TrashNetDataset(train_hf, transform=train_transform, cache_images=cache_images)
    val_dataset = TrashNetDataset(val_hf, transform=val_transform, cache_images=cache_images)
    test_dataset = TrashNetDataset(test_hf, transform=val_transform, cache_images=cache_images)
    print(f"  [DataLoader] Datasets created successfully")

    # Common dataloader kwargs for parallel optimization
    common_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,  # Always pin memory for GPU acceleration
        'persistent_workers': persistent_workers if num_workers > 0 else False,
    }

    # Add prefetch_factor only if num_workers > 0
    if num_workers > 0 and prefetch_factor is not None:
        common_kwargs['prefetch_factor'] = prefetch_factor

    print(f"  [DataLoader] Configuration:")
    print(f"    Batch size: {batch_size}")
    print(f"    Num workers: {num_workers}")
    print(f"    Pin memory: {common_kwargs['pin_memory']}")
    if num_workers > 0:
        print(f"    Prefetch factor: {prefetch_factor if prefetch_factor else 2}")
        print(f"    Persistent workers: {persistent_workers}")

    # Create dataloaders with parallel optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs
    )
    print(f"  [DataLoader] Train loader created: {len(train_loader)} batches")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs
    )
    print(f"  [DataLoader] Validation loader created: {len(val_loader)} batches")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs
    )
    print(f"  [DataLoader] Test loader created: {len(test_loader)} batches")

    print(f"\n[DataLoader] Summary:")
    print(f"  Training:   {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Test:       {len(test_dataset)} samples ({len(test_loader)} batches)")
    print(f"  Batch size: {batch_size}")
    if num_workers > 0:
        total_prefetch = num_workers * (prefetch_factor if prefetch_factor else 2)
        print(f"  Total prefetch capacity: {total_prefetch} batches ({total_prefetch * batch_size} samples)")

    return train_loader, val_loader, test_loader
