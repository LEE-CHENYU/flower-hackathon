import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional


class PrivacyProtectionModel(nn.Module):
    def __init__(self, privacy_budget: float = 1.0, blur_strength: int = 31):
        super(PrivacyProtectionModel, self).__init__()
        self.privacy_budget = privacy_budget
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.privacy_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        self.face_cascade = None
        self._init_face_detector()

    def _init_face_detector(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            print("Warning: Face detector not available, will apply general blur")

    def detect_and_blur_faces(self, image_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = image_tensor.shape[0]
        blurred_batch = []

        for i in range(batch_size):
            img = image_tensor[i].cpu()

            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            if self.face_cascade is not None:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    face_region = img_np[y:y+h, x:x+w]
                    face_region = cv2.GaussianBlur(face_region,
                                                   (self.blur_strength, self.blur_strength), 0)
                    img_np[y:y+h, x:x+w] = face_region
            else:
                img_np = cv2.GaussianBlur(img_np, (self.blur_strength, self.blur_strength), 0)

            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            blurred_batch.append(img_tensor)

        return torch.stack(blurred_batch)

    def add_differential_privacy(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            sensitivity = 1.0
            scale = sensitivity / self.privacy_budget

            noise_shape = features.shape
            noise = torch.normal(mean=0, std=scale, size=noise_shape,
                                  device=features.device)

            features = features + noise

        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_blurred = self.detect_and_blur_faces(x)

        encoded = self.encoder(x_blurred)

        private_features = self.privacy_layer(encoded)
        private_features = self.add_differential_privacy(private_features)

        reconstructed = self.decoder(private_features)

        return reconstructed, private_features

    def get_privacy_metrics(self) -> dict:
        return {
            'privacy_budget': self.privacy_budget,
            'blur_strength': self.blur_strength,
            'differential_privacy_enabled': True
        }


class PrivacyLoss(nn.Module):
    def __init__(self, reconstruction_weight: float = 0.5,
                 privacy_weight: float = 0.5):
        super(PrivacyLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.privacy_weight = privacy_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, original: torch.Tensor,
                reconstructed: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        reconstruction_loss = self.mse_loss(reconstructed, original)

        privacy_loss = -torch.mean(torch.std(features, dim=0))

        total_loss = (self.reconstruction_weight * reconstruction_loss +
                      self.privacy_weight * privacy_loss)

        return total_loss, reconstruction_loss, privacy_loss


def test_privacy_model():
    model = PrivacyProtectionModel(privacy_budget=1.0, blur_strength=31)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output, features = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Privacy metrics: {model.get_privacy_metrics()}")

    return model


if __name__ == "__main__":
    test_privacy_model()