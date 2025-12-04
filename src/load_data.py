from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config import load_config

config = load_config()

class TrashNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image']
        label = item['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image, label

def load_data(split_data=True):
    """
    Loads Trashnet dataset and either returns full dataset
    or split dataset based on train/val/test ratios in config.
    
    :param split_data: Whether or not to split the dataset.
    """
    dataset_name = config['data']['dataset_name']

    ds = load_dataset(dataset_name)
    full_ds = ds['train']

    if not split_data:
        return full_ds

    train_ratio = config['data']['train_split']
    val_ratio = config['data']['val_split']
    test_ratio = config['data']['test_split']

    train_val_split = full_ds.train_test_split(test_size=(val_ratio+test_ratio), seed=config['project']['random_seed'])
    train_dataset = train_val_split['train']

    val_test_split = train_val_split['test'].train_test_split(test_size=test_ratio/(val_ratio+test_ratio), seed=config['project']['random_seed'])
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    return train_dataset, val_dataset, test_dataset

def get_class_dist(dataset):
    from collections import Counter

    labels = dataset['label']
    label_names = dataset.features['label'].names
    label_counts = Counter(labels)

    return {label_names[i]: label_counts[i] for i in range(len(label_names))}