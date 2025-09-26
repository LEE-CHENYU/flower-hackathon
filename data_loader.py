import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import List, Tuple, Dict


class FirstPhotoDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else self.get_default_transform()
        self.image_paths = self._collect_first_photos()

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _collect_first_photos(self) -> List[Tuple[str, str]]:
        first_photos = []

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.root_dir, split)
            if not os.path.exists(split_dir):
                continue

            for person_folder in sorted(os.listdir(split_dir)):
                person_path = os.path.join(split_dir, person_folder)
                if not os.path.isdir(person_path):
                    continue

                for date_folder in sorted(os.listdir(person_path)):
                    date_path = os.path.join(person_path, date_folder)
                    if not os.path.isdir(date_path):
                        continue

                    jpg_files = [f for f in os.listdir(date_path)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    if jpg_files:
                        first_photo = sorted(jpg_files)[0]
                        full_path = os.path.join(date_path, first_photo)
                        folder_id = f"{split}/{person_folder}/{date_folder}"
                        first_photos.append((full_path, folder_id))

        print(f"Found {len(first_photos)} first photos across all folders")
        return first_photos

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, folder_id = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, folder_id, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 224, 224)), folder_id, img_path


def create_federated_dataloaders(root_dir: str, num_clients: int = 5,
                                  batch_size: int = 16) -> Dict[int, DataLoader]:
    dataset = FirstPhotoDataset(root_dir)

    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    client_dataloaders = {}
    indices = list(range(total_samples))
    np.random.shuffle(indices)

    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        if client_id == num_clients - 1:
            client_indices = indices[start_idx:]
        else:
            client_indices = indices[start_idx:start_idx + samples_per_client]

        client_dataset = torch.utils.data.Subset(dataset, client_indices)
        client_dataloaders[client_id] = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print(f"Client {client_id}: {len(client_dataset)} samples")

    return client_dataloaders


if __name__ == "__main__":
    root_dir = "/Users/chenyusu/flower-hackathon/omni_coco"
    dataset = FirstPhotoDataset(root_dir)
    print(f"Total dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample_img, folder_id, path = dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample folder ID: {folder_id}")
        print(f"Sample path: {path}")