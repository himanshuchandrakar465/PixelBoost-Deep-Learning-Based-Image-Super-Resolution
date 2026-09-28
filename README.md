<div align="center">

# 🔍 PixelBoost

### Deep Learning Image De-Pixelation with a Residual U-Net

Turn blocky, pixelated images back into clear, detailed ones.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)

</div>

---

## ✨ Overview

**PixelBoost** is a deep learning project that repairs pixelated images.
It takes a blocky image as input and gives back a cleaner image of the **same size**.

The model is a **Residual U-Net**. It learns only the *correction* and adds it to the input image. This makes training faster and more stable.

> 📝 **Note:** This project fixes pixelation (de-pixelation). It does not make the image bigger.

---

## 🖼️ Results

<!-- Add your result images to the images/ folder, then replace the name below -->
<div align="center">

| Pixelated | Model Output | Original |
|:---:|:---:|:---:|
| ![pixelated](images/pixelated.png) | ![output](images/output.png) | ![original](images/original.png) |

</div>

---

## 🚀 Features

- 🧱 **Residual U-Net** with 4 levels and skip connections
- 🔁 **Input + correction** output for stable learning
- 📦 **Patch training** (128×128) for low memory use
- 🔄 **Data augmentation:** flips and rotations
- 🎚️ **Random pixel strength:** 2×, 4×, 8×, 16×
- 📉 **Loss:** L1 + SSIM for sharper results
- ⏱️ **Smart training:** learning rate reduction, early stopping, best-model saving
- 📊 **Full evaluation:** MSE, MAE, RMSE, PSNR, SSIM, MS-SSIM
- 🖼️ **Any image size** at test time (input shape is `None, None, 3`)

---

## 🧠 How It Works

```
High-quality image
        │
        ▼
 Random 128×128 patch  ──►  Flip / Rotate
        │
        ▼
 Pixelate (shrink, then enlarge)
        │
        ▼
   Residual U-Net  ──►  Correction
        │
        ▼
  Input + Correction  =  Clean image
```

**Input shape:** `(H, W, 3)`
**Output shape:** `(H, W, 3)` (same as input)
**Pixel values:** 0 to 1

> ⚠️ Height and width must be divisible by **16** (4 pooling levels). The code adds a small border and removes it after prediction.

---

## 📂 Project Structure

```
PixelBoost/
├── images/                 # Result images for the README
├── src/                    # Source package
├── main.py                 # Main script
├── model_creation.py       # Model building code
├── preprosses.py           # Data loading and pixelation
├── import.py               # Imports and setup
├── __notebook_source__.ipynb   # Training notebook
├── resulation.ipynb        # Testing and evaluation notebook
├── best_model.keras        # Best saved model
├── last_model.keras        # Last saved model
├── requirements.txt        # Packages
├── pyproject.toml          # Project settings
└── README.md
```

---

## 📊 Dataset

**DIV2K** (high-quality 2K images)

- Training: 800 images
- Validation: 100 images
- Source: [DIV2K on Kaggle](https://www.kaggle.com/datasets/joe1995/div2k-dataset)

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/himanshuchandrakar465/PixelBoost-Deep-Learning-Based-Image-Super-Resolution.git
cd PixelBoost-Deep-Learning-Based-Image-Super-Resolution
```

2. Install the packages:

```bash
pip install -r requirements.txt
```

3. **Easiest way:** open the notebook in **Google Colab** and turn on the **GPU**
(Runtime → Change runtime type → GPU).

---

## ▶️ Usage

### Train

Open `__notebook_source__.ipynb` and run the cells in order.

### Test and evaluate

Open `resulation.ipynb`. It loads the saved model and compares the pixelated image, the model output, and the original.

### Load the saved model

```python
import tensorflow as tf

model = tf.keras.models.load_model("best_model.keras", compile=False)
```

### Predict on one image

```python
import numpy as np

def predict_image(model, img):
    h, w, _ = img.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    out = model.predict(padded[None, ...], verbose=0)[0]
    return np.clip(out[:h, :w], 0, 1)
```

---

## 📏 Metrics

| Metric | Better when | Meaning |
|---|---|---|
| **MSE** | Lower | Average squared pixel error |
| **MAE** | Lower | Average absolute pixel error |
| **RMSE** | Lower | Square root of MSE |
| **PSNR** | Higher | Signal quality in dB (over 30 dB is good) |
| **SSIM** | Higher | Structure similarity (1.0 is perfect) |
| **MS-SSIM** | Higher | SSIM at many scales |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## 🔮 Future Work

- [ ] Pix2Pix GAN with a PatchGAN discriminator
- [ ] VGG perceptual loss for sharper detail
- [ ] Tile-based prediction for very large images
- [ ] Real-world image damage (noise, blur, compression)
- [ ] Simple web app demo

---

## 🤝 Contributing

Ideas and pull requests are welcome. Open an issue to start.

---

## 🙏 Acknowledgements

- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- U-Net: Ronneberger et al., 2015
- TensorFlow and Keras teams

---

<div align="center">

Made by **Himanshu Chandrakar**

⭐ If you like this project, give it a star!

</div>
