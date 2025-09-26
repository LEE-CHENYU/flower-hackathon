"""
Simple data loader for flat directory structure
"""

import os
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Simple dataset for loading images from flat directories

        Args:
            root_dir: Root directory containing train/val/test folders with images
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else self.get_default_transform()
        self.image_paths = self._collect_images()

        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _collect_images(self) -> List[Tuple[str, str]]:
        """Collect all images from train/val/test directories"""
        images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for split in ['train', 'val', 'test']:
            split_dir = self.root_dir / split

            if not split_dir.exists():
                continue

            # Get all image files in the directory
            for file_path in split_dir.iterdir():
                if file_path.suffix.lower() in image_extensions:
                    images.append((str(file_path), split))

        return sorted(images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, split = self.image_paths[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'path': img_path,
            'split': split
        }