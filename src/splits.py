import json
import numpy as np
from pathlib import Path
from src.load_data import get_class_dist

def save_splits(train_dataset, val_dataset, test_dataset, config, splits_dir=None):
    if splits_dir is None:
        root = Path(__file__).parent.parent
        splits_dir = root / config['paths']['splits_dir']
    else:
        splits_dir = Path(splits_dir)

    splits_dir.mkdir(parents=True, exist_ok=True)

    train_index = np.array(range(len(train_dataset)))
    val_index = np.array(range(len(val_dataset)))
    test_index = np.array(range(len(test_dataset)))
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)

    with open(splits_dir / 'train_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, train_index)))
    with open(splits_dir / 'val_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, val_index)))
    with open(splits_dir / 'test_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, test_index)))

    train_dist = get_class_dist(train_dataset)
    val_dist = get_class_dist(val_dataset)
    test_dist = get_class_dist(test_dataset)

    splits_meta = {
        'train_indices': train_index.tolist(),
        'val_indices': val_index.tolist(),
        'test_indices': test_index.tolist(),
        'seed': config['project']['random_seed'],
        'total': total,
        'train_size': len(train_index),
        'val_size': len(val_index),
        'test_size': len(test_index),
        'train_ratio': config['data']['train_split'],
        'val_ratio': config['data']['val_split'],
        'test_ratio': config['data']['test_split'],
        'dataset_name': config['data']['dataset_name'],
        'train_class_dist': train_dist,
        'val_class_dist': val_dist,
        'test_class_dist': test_dist
    }
    with open(splits_dir / 'splits.json', 'w') as f:
        json.dump(splits_meta, f, indent=2)
    
    return splits_meta

def load_splits_meta(splits_dir):
    splits_dir = Path(splits_dir)

    with open(splits_dir / 'splits.json', 'r') as f:
        metadata = json.load(f)

    return metadata
