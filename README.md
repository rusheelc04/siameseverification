# Siamese Network for One-Shot Image Recognition

**By Rusheel Chande**

This project implements a facial verification (or “face ID”) system using a custom Siamese neural network trained to compare image pairs. By leveraging a *distance-based* approach (in this case, L1 distance) over learned deep embeddings, the model can determine whether two given images show the same person’s face.

---

## Overview

This project is centered on applying a **Siamese neural network** to the task of facial verification, especially in low-data or “one-shot” contexts. The core objective is to determine whether two input images depict the same identity. Rather than conventional classification (where many examples per class are necessary), the Siamese approach compares just two images at a time, making it well-suited to “one-shot learning” scenarios.

**Key capabilities and components include:**
- **Data augmentation** for robust training with a small number of face images.
- **Deep embedding architecture** that learns powerful, generalizable features for face matching.
- **Real-time verification** using OpenCV to capture frames, feeding them into the trained Siamese model.
- **Scalable thresholding** approach that allows fine-tuning the sensitivity (detection threshold) and overall verification strictness (verification threshold).

---

## Research Paper & Inspiration

This work draws heavily from ideas in the research paper:

**Koch, G., Zemel, R., & Salakhutdinov, R. (2015).** “Siamese Neural Networks for One-shot Image Recognition.”

The paper explores how Siamese networks can learn from limited examples (“one-shot” scenarios) by mapping images into a feature space where pairs belonging to the same class cluster more closely than pairs belonging to different classes. We adapt that methodology to real-world face verification.

---

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

### Siamese Distance Layer
After generating embeddings for each image, we use a custom `L1Dist` layer:

```python
class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
```

This absolute difference vector is fed into a final `Dense` layer to yield a probability indicating how likely the two embeddings (i.e., images) are of the same identity.

---

## Training Loop & Loss Function
- **Binary Cross-Entropy** is used for the verification task (output = 1 if same identity, 0 otherwise).
- **Adam optimizer** with a small learning rate is typically employed.
- **Checkpoints** are periodically saved for training continuity.

**Pseudocode** for the train step:
```python
with tf.GradientTape() as tape:
    # Forward pass (predict same/not same)
    yhat = siamese_model([anchor_imgs, positive_or_negative_imgs])
    # Compute loss
    loss = binary_cross_loss(true_labels, yhat)

# Backprop & update weights
grads = tape.gradient(loss, siamese_model.trainable_variables)
opt.apply_gradients(zip(grads, siamese_model.trainable_variables))
```

---

## Setup & Installation

### Dependencies
- **Python 3.7+**
- [**TensorFlow**](https://www.tensorflow.org/) (>=2.4)
- [**OpenCV**](https://opencv.org/) (>=4.0)
- **NumPy**, **Matplotlib**, etc.
- [**Kivy**](https://kivy.org/) for the optional real-time verification GUI

Install them via:
```bash
pip install --upgrade pip
pip install tensorflow opencv-python matplotlib kivy
```
### Folder Structure
```plaintext
.
├─ data
│   ├─ anchor        # Images from user (anchor)
│   ├─ positive      # Additional images of user (positive)
│   └─ negative      # Images of other people (negative)
├─ app
│   ├─ faceid.py     # The Kivy-based GUI application
│   ├─ layers.py     # Custom L1Dist layer
│   └─ application_data
│       ├─ input_image           # Where a new captured frame goes
│       └─ verification_images   # 50 reference images to compare
├─ SiameseNNForOneShotImageRecognitionResearchPaper.pdf
├─ Notebook.ipynb    # Full tutorial notebook
├─ siamesemodel.h5   # Trained model checkpoint
└─ README.md
```

---

## Usage

### Collecting Anchor/Positive Samples
1. **Run** the data collection cell in the Jupyter notebook or a standalone script.
2. Press **`a`** to capture anchor images of yourself.
3. Press **`p`** to capture positive samples (additional images of you).
4. Negative images typically come from a dataset like [LFW](http://vis-www.cs.umass.edu/lfw/) or your own curated set.

### Training
To train via the included **Notebook.ipynb**:

1. **Run** the *Install Dependencies* cell (if needed).
2. **Create folder structures** (`anchor`, `positive`, `negative`) under `data/`.
3. *(Optional)* **Data augmentation** to expand your training data.
4. **Train** the model:
   ```python
   EPOCHS = 50
   train(train_data, EPOCHS)
   ```
### Evaluation & Testing
Use the prepared test partition or manually collect a new set of pairs. The notebook demonstrates:

- **Precision & Recall** metrics on unseen data.
- Fine-tuning **detection thresholds**.

### Real-Time Face Verification
We provide a sample **Kivy app** (`faceid.py`) that:
- Accesses your webcam in real time.
- Captures frames upon pressing **Verify**.
- Preprocesses the captured frame and compares it against verification images.
- Displays **Verified** or **Unverified** in the app window.

**To launch:**
```bash
cd app
python faceid.py
```
A window will open with:

- **Webcam feed** at the top  
- **Verify button** to trigger the Siamese comparison  
- **Label** to display the verification status  

---

## Project Highlights & Possible Improvements

### Strengths
- **One-Shot Learning**: Even with minimal user images, the method can generalize well, thanks to the learned embedding space.  
- **Modular Code**: The notebook cleanly separates data loading, model definition, and training loops.  
- **Real-Time Demo**: Quick approach to “Face ID” using your webcam.

### Potential Enhancements
1. **Advanced Data Augmentation**: Beyond global affine transformations, incorporate more domain-specific augmentations (e.g., random occlusions, face alignment).  
2. **Larger Negative Dataset**: More variety in the “negative” category to reduce false positives.  
3. **Hyperparameter Tuning**: Automate or further refine learning rates, optimizer parameters, batch sizes, etc.  
4. **Adaptation for Multi-Face**: Extend beyond 1:1 verification to multi-person classification or tracking.  
5. **Use Transfer Learning**: If desired, adopt a pretrained face recognition network as the embedding backbone.

---

## Repository Contents

- **Notebook.ipynb**: End-to-end code for data collection, training, and evaluation in a step-by-step format.  
- **app/faceid.py**: A Kivy GUI app for real-time face verification using your webcam.  
- **app/layers.py**: Custom L1 distance layer required by the Siamese model.  
- **data/**: Contains subfolders (`anchor`, `positive`, `negative`) for face samples.  
- **app/application_data/**: Contains subfolders for storing an input image on-the-fly and a set of verification images.

---

## Acknowledgments

- **Gregory Koch, Richard Zemel, Ruslan Salakhutdinov** for their paper on *Siamese Neural Networks for One-shot Image Recognition*.
- **Kivy** to build cross-platform Python apps with a visual interface.
- **TensorFlow** for the deep learning ecosystem.
- The many open-source contributors who maintain libraries like NumPy, OpenCV, Matplotlib, etc.
