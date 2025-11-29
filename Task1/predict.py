import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SimpleCnn(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_transform() -> transforms.Compose:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = SimpleCnn(num_classes=len(CIFAR10_CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: nn.Module, image_path: Path, device: torch.device) -> str:
    tfm = build_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    score, idx = probs.max(dim=1)
    label = CIFAR10_CLASSES[idx.item()]
    return f"{image_path.name}: {label} ({score.item():.2f})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a trained CIFAR-10 model")
    p.add_argument("input", type=str, help="Path to an image file or directory of images")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/cifar10_cnn.pth",
        help="Path to model checkpoint (.pth)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = load_model(ckpt, device)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = [p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not images:
            raise FileNotFoundError(f"No images found in folder: {input_path}")
        for img in images:
            print(predict_image(model, img, device))
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Image not found: {input_path}")
        print(predict_image(model, input_path, device))


if __name__ == "__main__":
    main()


