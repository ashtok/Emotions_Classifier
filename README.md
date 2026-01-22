# Facial Emotion Recognition CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-84.24%25-brightgreen)](README.md)

A lightweight CNN trained on 59,099 facial images to classify 5 emotions: Angry, Fear, Happy, Sad, and Surprise.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **84.24%** |
| Val Accuracy | **83.94%** |
| Training Time | ~2 hours (100 epochs) |
| Dataset Split | 70% train / 15% val / 15% test |

## Model Architecture

```
EmotionClassify_CNN(
  conv1: Conv2d(3 → 32, 3×3, padding=1)
  conv2: Conv2d(32 → 64, 3×3, padding=1)
  pool: MaxPool2d(2×2)
  fc1: Linear(65536 → 128)
  fc2: Linear(128 → 5)
  dropout: Dropout(p=0.15)
)
```

**Input:** 128×128 RGB images  
**Output:** 5-class emotion predictions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition

# Install dependencies
pip install torch torchvision numpy pandas matplotlib tqdm

# Run the notebook
jupyter notebook classify-human-emotions.ipynb
```

The trained model weights are saved in `emotion_cnn.pth`.

## Training Details

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Batch Size:** 64
- **Image Transforms:** Resize to 128×128, ToTensor
- **Hardware:** GPU-accelerated (CUDA)

## Dataset

- **Total Images:** 59,099
- **Classes:** Angry (0), Fear (1), Happy (2), Sad (3), Surprise (4)
- **Source:** Human Face Emotions dataset

## Key Features

- Simple 2-layer CNN architecture
- Fast training (~74s per epoch on GPU)
- Minimal preprocessing required
- Model checkpoint saving/loading included
