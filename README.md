# Siamese Network for One-Shot Image Recognition

**By Rusheel Chande**

This project implements a facial verification (or “face ID”) system using a custom Siamese neural network trained to compare image pairs. By leveraging a *distance-based* approach (in this case, L1 distance) over learned deep embeddings, the model can determine whether two given images show the same person’s face.

## Overview

This project is centered on applying a **Siamese neural network** to the task of facial verification, especially in low-data or “one-shot” contexts. The core objective is to determine whether two input images depict the same identity. Rather than conventional classification (where many examples per class are necessary), the Siamese approach compares just two images at a time, making it well-suited to “one-shot learning” scenarios.

**Key capabilities and components include:**
- **Data augmentation** for robust training with a small number of face images.
- **Deep embedding architecture** that learns powerful, generalizable features for face matching.
- **Real-time verification** using OpenCV to capture frames, feeding them into the trained Siamese model.
- **Scalable thresholding** approach that allows fine-tuning the sensitivity (detection threshold) and overall verification strictness (verification threshold).

## Research Paper & Inspiration

This work draws heavily from ideas in the research paper:

**Koch, G., Zemel, R., & Salakhutdinov, R. (2015).** “Siamese Neural Networks for One-shot Image Recognition.”

The paper explores how Siamese networks can learn from limited examples (“one-shot” scenarios) by mapping images into a feature space where pairs belonging to the same class cluster more closely than pairs belonging to different classes. We adapt that methodology to real-world face verification.

## How It Works

### Data Collection & Preprocessing

- **Face Images (Anchor/Positive/Negative)**
  - **Anchor images:** reference images of the target user.
  - **Positive images:** additional images of the same user.
  - **Negative images:** images of other individuals’ faces (e.g., from the LFW dataset).

- **Preprocessing**
  - All images are resized to **100×100**.
  - Pixel values are normalized to **[0,1]**.
  - **Optional augmentation** (random brightness/contrast flips, etc.) is used to improve generalization.

### Embedding Network Architecture
We use a convolutional “embedding” sub-network that converts an input image into a **4096-dimensional feature vector**:

- **Convolution + ReLU + MaxPooling** repeated multiple times.
- **Flatten** the final convolution output.
- **Dense(4096) with Sigmoid** for the final embedding.

**Output**: A high-level feature vector representing the input image.

<details>
<summary>Example Architecture</summary>

```plaintext
Input (100x100x3)
 ┃
 ┣━ Conv2D(64, kernel=10x10, ReLU) -> MaxPool2D
 ┃
 ┣━ Conv2D(128, kernel=7x7, ReLU) -> MaxPool2D
 ┃
 ┣━ Conv2D(128, kernel=4x4, ReLU) -> MaxPool2D
 ┃
 ┣━ Conv2D(256, kernel=4x4, ReLU)
 ┃
 ┗━ Flatten -> Dense(4096, activation='sigmoid')
       ┗━ Output: 4096-D embedding
```
</details>

