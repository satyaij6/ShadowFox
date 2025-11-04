# Image Classification on CIFAR-10 (PyTorch)

This project trains a simple CNN classifier on CIFAR-10 using PyTorch. It can automatically use your local `cifar-10-python.tar.gz` (placed in the project root) or download the dataset from here "https://www.cs.toronto.edu/~kriz/cifar.html".

## Quick Start (Windows PowerShell)

```
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# PowerShell activation (either of the following works):
# Option A (dot-source with space):
. .\.venv\Scripts\Activate.ps1
# Option B (directly execute the script):
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train (uses local cifar-10 archive if present; otherwise downloads)
python train.py --epochs 10 --batch-size 128 --lr 1e-3 \
  --data-dir data \
  --archive-path cifar-10-python.tar.gz \
  --save-path outputs/cifar10_cnn.pth
```

- The best model checkpoint is saved to `outputs/cifar10_cnn.pth`.
- By default, training uses GPU if available; otherwise CPU.

## Arguments
- `--epochs`: number of training epochs (default: 10)
- `--batch-size`: batch size (default: 128)
- `--lr`: learning rate (default: 1e-3)
- `--data-dir`: directory to store/extract CIFAR-10 (default: `data`)
- `--archive-path`: path to local `cifar-10-python.tar.gz` (default: `cifar-10-python.tar.gz`)
- `--num-workers`: DataLoader workers (default: 2)
- `--seed`: random seed (default: 42)
- `--save-path`: where to save the best model (default: `outputs/cifar10_cnn.pth`)

## Notes
- If `data/cifar-10-batches-py` exists, the script will use it.
- If the local archive exists, it will be extracted automatically.
- If neither exists, the script will let `torchvision` download the dataset.
 

## Run inference (tag images)

```
# Single image
python predict.py path\to\image.jpg --checkpoint outputs\cifar10_cnn.pth

# All images in a folder
python predict.py path\to\images_folder --checkpoint outputs\cifar10_cnn.pth
```

Output format: `filename: label (probability)`

## Troubleshooting PowerShell activation
- If you see "The module '.venv' could not be loaded" or an execution policy error, run this once in the same PowerShell session, then activate again:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

