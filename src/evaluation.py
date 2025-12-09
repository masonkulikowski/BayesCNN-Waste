"""
Evaluation utilities for CNN models.

This module contains functions for model evaluation and hyperparameter tuning.
"""

import torch
from tqdm.auto import tqdm
from pathlib import Path
import json
from itertools import product
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_model(model, test_loader, device, model_name='Model'):
    """
    Evaluate model on test set.

    Args:
        model: PyTorch model
        test_loader: Test dataloader
        device: Device to evaluate on
        model_name: Name of the model for display

    Returns:
        Tuple of (accuracy, predictions, labels, confusion_matrix)
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Testing {model_name}'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{model_name} Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return accuracy, all_preds, all_labels, cm


def hyperparameter_search(model_fn, param_grid, train_loader, val_loader, config, model_base_name='model', device='cpu'):
    """
    Perform grid search over hyperparameters.

    Args:
        model_fn: Function that creates the model
        param_grid: Dictionary of hyperparameters to search
        train_loader: Training dataloader (or None to create inside)
        val_loader: Validation dataloader (or None to create inside)
        config: Configuration dictionary
        model_base_name: Base name for saving models
        device: Device to train on

    Returns:
        results: List of tuning results
    """
    from src.training import train_model, create_dataloaders
    from src.load_data import load_data
    from src.transforms import get_transforms

    results = []

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_runs = 1
    for values in param_values:
        total_runs *= len(values)

    print(f"\n{'='*80}")
    print(f"Starting Hyperparameter Search")
    print(f"Total configurations to test: {total_runs}")
    print(f"Parameters: {list(param_grid.keys())}")
    print(f"{'='*80}\n")

    run_num = 0
    for param_combo in product(*param_values):
        run_num += 1
        params = dict(zip(param_names, param_combo))

        print(f"\n{'='*80}")
        print(f"Configuration {run_num}/{total_runs}")
        print(f"Parameters: {params}")
        print(f"{'='*80}\n")

        # Update config
        config_copy = config.copy()
        for key, value in params.items():
            if key in config_copy['training']:
                config_copy['training'][key] = value

        # Create dataloaders with new batch size if needed
        if 'batch_size' in params:
            train_hf, val_hf, test_hf = load_data(split_data=True)

            config_aug = config_copy.copy()
            config_aug['augmentation']['enabled'] = True
            train_transform = get_transforms(config_aug, split='train')
            val_transform = get_transforms(config_copy, split='val')

            train_loader_temp, val_loader_temp, _ = create_dataloaders(
                train_hf, val_hf, test_hf,
                train_transform, val_transform,
                batch_size=params['batch_size']
            )
        else:
            train_loader_temp, val_loader_temp = train_loader, val_loader

        # Create model
        model = model_fn()

        # Create model name
        param_str = '_'.join([f"{k}{v}" for k, v in params.items()])
        model_name = f"{model_base_name}_{param_str}"

        # Train
        try:
            history = train_model(
                model,
                train_loader_temp,
                val_loader_temp,
                config_copy,
                model_name=model_name,
                device=device,
                save_best=True
            )

            # Save results
            result = {
                'params': params,
                'best_val_acc': max(history['val_acc']),
                'best_val_loss': min(history['val_loss']),
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'epochs_trained': len(history['train_loss'])
            }
            results.append(result)

            print(f"\n✓ Result: Best Val Acc = {result['best_val_acc']:.2f}%")

        except Exception as e:
            print(f"✗ Error training with params {params}: {e}")
            continue

    # Print summary
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search Complete")
    print(f"{'='*80}\n")

    # Sort by best validation accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    print("\nTop 5 Configurations:")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['params']}")
        print(f"   Best Val Acc: {result['best_val_acc']:.2f}% | Val Loss: {result['best_val_loss']:.4f} | Epochs: {result['epochs_trained']}")
        print()

    # Save results
    results_path = Path(config['paths']['results_dir']) / f"{model_base_name}_tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {results_path}")

    return results


def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load model from checkpoint.

    Args:
        model: PyTorch model (architecture)
        checkpoint_path: Path to checkpoint file
        device: Device to load to (can be string or torch.device)

    Returns:
        model: Loaded model
    """
    # Load to CPU first to avoid DirectML device issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # Then move to target device
    model.to(device)

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model
