import argparse
import os
import random
import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_cifar10_available(data_dir: Path, archive_path: Path | None) -> None:
    """Ensure CIFAR-10 python version exists under data_dir.

    Priority:
    1) If extracted folder exists, do nothing.
    2) Else if local archive provided/found, extract it.
    3) Else allow torchvision to download later.
    """
    extracted_dir = data_dir / "cifar-10-batches-py"
    if extracted_dir.exists():
        return

    if archive_path is None:
        # Nothing to extract; downloading will be attempted by torchvision.
        return

    if not archive_path.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(data_dir)


class SimpleCnn(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
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


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    allow_download: bool,
) -> Tuple[DataLoader, DataLoader]:
    # Standard CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.CIFAR10(
        root=str(data_dir), train=True, transform=train_tfms, download=allow_download
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, transform=test_tfms, download=allow_download
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total += images.size(0)
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


def train(
    epochs: int,
    batch_size: int,
    lr: float,
    data_dir: Path,
    archive_path: Path | None,
    num_workers: int,
    seed: int,
    save_path: Path,
):
    set_seed(seed)

    # Prepare data
    ensure_cifar10_available(data_dir, archive_path)
    extracted_dir = data_dir / "cifar-10-batches-py"
    allow_download = not extracted_dir.exists()

    train_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        allow_download=allow_download,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model/opt
    model = SimpleCnn(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        running_loss = 0.0
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"Saved new best model to: {save_path}")

    print(f"Best accuracy: {best_acc*100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CIFAR-10 classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store/extract CIFAR-10",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default="cifar-10-python.tar.gz",
        help="Path to local cifar-10-python.tar.gz (optional)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/cifar10_cnn.pth",
        help="Where to save the best model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    archive_path: Path | None
    if args.archive_path:
        archive_path = Path(args.archive_path)
    else:
        archive_path = None

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=data_dir,
        archive_path=archive_path,
        num_workers=args.num_workers,
        seed=args.seed,
        save_path=Path(args.save_path),
    )


if __name__ == "__main__":
    main()


