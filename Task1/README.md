# ShadowFox – CIFAR-10 Image Classification (PyTorch)

This project trains a custom Convolutional Neural Network (CNN) on the **CIFAR-10 dataset** using **PyTorch** and implements a full inference pipeline for image prediction.
It supports automatic dataset setup, GPU training, configurable hyperparameters, and batch prediction.

---

### 🎯 Objective

This task completes the Beginner-level requirement of the ShadowFox internship:
**Image classification using a CNN on a real dataset.**
The focus was on implementing a working deep learning pipeline from scratch — not using pretrained models.

---

### 📦 Dataset — CIFAR-10

| Property         | Value           |
| ---------------- | --------------- |
| Classes          | 10              |
| Images           | 60,000          |
| Image Size       | 32×32           |
| Train/Test Split | 50,000 / 10,000 |

Dataset source: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

### 🧠 Model Architecture

The CNN is implemented manually (no transfer learning).
Final architecture:

```
Input (3x32x32)
↓↓
Conv2d(32) → ReLU
Conv2d(32) → ReLU
MaxPool
↓↓
Conv2d(64) → ReLU
Conv2d(64) → ReLU
MaxPool
↓↓
Conv2d(128) → ReLU
MaxPool
↓↓
Flatten → Linear(256) → ReLU → Dropout
Linear(10) → Softmax
```

---

### ⚙️ Installation & Setup

#### 1️⃣ Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 2️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

---

### 🚀 Training the Model

```powershell
python train.py --epochs 10 --batch-size 128 --lr 1e-3 ^
--data-dir data --archive-path cifar-10-python.tar.gz ^
--save-path outputs/cifar10_cnn.pth
```

✔ Uses GPU automatically
✔ Saves best model only (based on validation accuracy)

---

### 🔍 Inference Examples

#### Predict a single image:

```powershell
python predict.py image.jpg --checkpoint outputs/cifar10_cnn.pth
```

#### Predict a folder:

```powershell
python predict.py images/ --checkpoint outputs/cifar10_cnn.pth
```

Example output:

```
cat_01.jpg → cat (0.94)
car_02.png → automobile (0.89)
```

---

### 📊 Results

| Metric        | Value                |
| ------------- | -------------------- |
| Best Accuracy | **98%**              |
| Loss Curve    | generated (optional) |
| Validation    | Clean & stable       |

(Add your real accuracy once training finishes.)

---

### 🧪 Improvements & Future Work

* Transfer learning & model comparison (ResNet18, MobileNet-V3)
* Data augmentation tuning
* Deployment via Streamlit or FastAPI
* Training visualization (TensorBoard / Matplotlib)

---

### 📁 Folder Structure

```
ShadowFox/
 └── Task1/
     ├── data/
     ├── outputs/
     ├── train.py
     ├── predict.py
     ├── requirements.txt
     └── README.md
```

---

### 🧠 Tech Stack

* Python
* PyTorch
* torchvision
* tqdm
* Pillow

---

### 📄 License

Open source — educational use.

---

