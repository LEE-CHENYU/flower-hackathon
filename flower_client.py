import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
from collections import OrderedDict

from privacy_model import PrivacyProtectionModel, PrivacyLoss
from data_loader import FirstPhotoDataset


class PrivacyProtectionClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, dataloader: DataLoader,
                 privacy_budget: float = 1.0, blur_strength: int = 30):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PrivacyProtectionModel(
            privacy_budget=privacy_budget,
            blur_strength=blur_strength
        ).to(self.device)

        self.criterion = PrivacyLoss(
            reconstruction_weight=0.5,
            privacy_weight=0.5
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.train(epochs=config.get("epochs", 1))
        return self.get_parameters(config={}), len(self.dataloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, privacy_score = self.test()
        return float(loss), len(self.dataloader.dataset), {"privacy_score": float(privacy_score)}

    def train(self, epochs: int = 1) -> None:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (images, folder_ids, paths) in enumerate(self.dataloader):
                images = images.to(self.device)

                self.optimizer.zero_grad()

                reconstructed, features = self.model(images)

                loss, recon_loss, privacy_loss = self.criterion(
                    images, reconstructed, features
                )

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                total_samples += images.size(0)

                if batch_idx % 10 == 0:
                    print(f"Client {self.client_id} - Epoch {epoch+1}, "
                          f"Batch {batch_idx}/{len(self.dataloader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Recon: {recon_loss.item():.4f}, "
                          f"Privacy: {privacy_loss.item():.4f}")

            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"Client {self.client_id} - Epoch {epoch+1} completed. "
                  f"Avg Loss: {avg_epoch_loss:.4f}, "
                  f"Samples processed: {total_samples}")

    def test(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_privacy_score = 0.0
        batch_count = 0

        with torch.no_grad():
            for images, folder_ids, paths in self.dataloader:
                images = images.to(self.device)

                reconstructed, features = self.model(images)

                loss, recon_loss, privacy_loss = self.criterion(
                    images, reconstructed, features
                )

                total_loss += loss.item()

                privacy_score = torch.std(features).item()
                total_privacy_score += privacy_score
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_privacy = total_privacy_score / batch_count if batch_count > 0 else 0

        print(f"Client {self.client_id} - Test Loss: {avg_loss:.4f}, "
              f"Privacy Score: {avg_privacy:.4f}")

        return avg_loss, avg_privacy


def create_client_fn(client_dataloaders: Dict[int, DataLoader],
                     privacy_budget: float = 1.0,
                     blur_strength: int = 30):
    def client_fn(cid: str) -> PrivacyProtectionClient:
        client_id = int(cid)
        dataloader = client_dataloaders[client_id]
        return PrivacyProtectionClient(
            client_id=client_id,
            dataloader=dataloader,
            privacy_budget=privacy_budget,
            blur_strength=blur_strength
        )

    return client_fn


if __name__ == "__main__":
    from data_loader import create_federated_dataloaders

    root_dir = "/Users/chenyusu/flower-hackathon/omni_coco"
    client_dataloaders = create_federated_dataloaders(root_dir, num_clients=3, batch_size=8)

    client = PrivacyProtectionClient(
        client_id=0,
        dataloader=client_dataloaders[0],
        privacy_budget=1.0,
        blur_strength=30
    )

    print(f"Client 0 initialized with {len(client_dataloaders[0].dataset)} samples")