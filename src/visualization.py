"""
Visualization utilities for CNN training and evaluation.

This module contains functions for plotting training curves and saving results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def plot_training_curves(history, model_name='Model', save_dir='results'):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary containing training history
        model_name: Name of the model for plot title
        save_dir: Directory to save plots

    Returns:
        None (saves plot to file and displays)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Mark best validation loss
    best_val_idx = np.argmin(history['val_loss'])
    axes[0].plot(best_val_idx + 1, history['val_loss'][best_val_idx], 'r*', markersize=15, label='Best')

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Mark best validation accuracy
    best_val_idx = np.argmax(history['val_acc'])
    axes[1].plot(best_val_idx + 1, history['val_acc'][best_val_idx], 'r*', markersize=15, label='Best')

    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir) / f"{model_name.lower().replace(' ', '_')}_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")

    plt.show()


def save_training_history(history, model_name='model', save_dir='results'):
    """
    Save training history to JSON.

    Args:
        history: Dictionary containing training history
        model_name: Name of the model
        save_dir: Directory to save results

    Returns:
        None (saves JSON file)
    """
    save_path = Path(save_dir) / f"{model_name}_history.json"
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {save_path}")


def plot_comparison(results_dict, metric='val_acc', save_dir='results'):
    """
    Plot comparison of multiple models.

    Args:
        results_dict: Dictionary of {model_name: history}
        metric: Metric to compare ('val_acc' or 'val_loss')
        save_dir: Directory to save plots

    Returns:
        None (saves plot to file and displays)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, history in results_dict.items():
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], label=model_name, linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ylabel = 'Accuracy (%)' if metric == 'val_acc' else 'Loss'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Model Comparison - Validation {ylabel}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir) / f"model_comparison_{metric}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")

    plt.show()


def print_model_summary(model, model_name='Model'):
    """
    Print a summary of model architecture and parameters.

    Args:
        model: PyTorch model
        model_name: Name of the model

    Returns:
        None (prints to console)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*60}")
    print(f"{model_name} Architecture Summary")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")
    print(f"{'='*60}\n")


def plot_hyperparameter_results(results, param_name='learning_rate', save_dir='results'):
    """
    Plot hyperparameter tuning results.

    Args:
        results: List of tuning results from hyperparameter_search
        param_name: Parameter to plot on x-axis
        save_dir: Directory to save plots

    Returns:
        None (saves plot to file and displays)
    """
    # Extract data
    param_values = [r['params'][param_name] for r in results]
    val_accs = [r['best_val_acc'] for r in results]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_values, val_accs, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'Hyperparameter Tuning: {param_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mark best
    best_idx = np.argmax(val_accs)
    ax.plot(param_values[best_idx], val_accs[best_idx], 'r*', markersize=20, label='Best')
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir) / f"hyperparameter_{param_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Hyperparameter plot saved to {save_path}")

    plt.show()
