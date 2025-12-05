import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def get_transforms(config, split='train'):
    image_size = config['data']['image_size']

    if split == 'train' and config['augmentation']['enabled']:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(config['augmentation']['rotation']),
        ]

        if config['augmentation']['horizontal_flip']:
            transform_list.append(transforms.RandomHorizontalFlip())

        if config['augmentation']['vertical_flip']:
            transform_list.append(transforms.RandomVerticalFlip())

        if config['augmentation']['brightness'] > 0 or config['augmentation']['contrast'] > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config['augmentation']['brightness'],
                    contrast=config['augmentation']['contrast']
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalization']['mean'],
                std=config['normalization']['std']
            )
        ])

    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalization']['mean'],
                std=config['normalization']['std']
            )
        ]

    return transforms.Compose(transform_list)


def get_base_transforms(config):
    image_size = config['data']['image_size']

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_normalization_transform(config):
    return transforms.Normalize(
        mean=config['normalization']['mean'],
        std=config['normalization']['std']
    )


def denormalize(tensor, config):
    mean = torch.tensor(config['normalization']['mean']).view(-1, 1, 1)
    std = torch.tensor(config['normalization']['std']).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def compute_dataset_stats(dataset, config, num_samples=None):
    print("Computing dataset statistics...")

    transform = get_base_transforms(config)

    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    mean = torch.zeros(3)
    std = torch.zeros(3)

    print(f"Processing {num_samples} images...")
    for i in range(num_samples):
        # HuggingFace datasets return dictionaries
        item = dataset[i]
        image = item['image'] if isinstance(item, dict) else item[0]

        if isinstance(image, Image.Image):
            image = transform(image)

        mean += image.mean(dim=[1, 2])
        std += image.std(dim=[1, 2])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{num_samples} images")

    mean /= num_samples
    std /= num_samples

    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

    print(f"\nDataset Statistics:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")

    return stats


def visualize_augmentations(image, config, num_samples=5):
    train_transform = get_transforms(config, split='train')

    augmented = []
    for _ in range(num_samples):
        aug_img = train_transform(image)
        aug_img = denormalize(aug_img, config)
        augmented.append(aug_img)

    return augmented


def tensor_to_image(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    np_img = (np_img * 255).astype(np.uint8)

    return Image.fromarray(np_img)

