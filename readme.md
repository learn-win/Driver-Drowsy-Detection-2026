# Video-Based Drowsiness Detection: The VBFLLFA Pipeline
**An End-to-End Deep Learning Architecture for Real-Time Fatigue Analysis**


## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [The Dataset (YawDD)](#-the-dataset-yawdd)
3. [Mathematical Feature Engineering](#-mathematical-feature-engineering)
4. [VBFL Model Architecture](#-vbfl-model-architecture)
5. [Pipeline Workflow](#-pipeline-workflow)
6. [Advanced Training Mechanics](#-advanced-training-mechanics)
7. [Installation & Setup](#-installation--setup)
8. [Directory Structure](#-directory-structure)
9. [Citation & References](#-citation--references)

---

## 📌 Project Overview
This repository implements a highly optimized, temporal-aware deep learning pipeline designed to detect driver fatigue. Unlike traditional computer vision models that rely on heavy 3D-Convolutional Neural Networks (3D-CNNs) to process raw video pixels, this pipeline utilizes a two-step approach:
1. **Spatial Extraction:** Lightweight mathematical facial metrics are extracted frame-by-frame.
2. **Temporal Modeling:** A custom Transformer-based neural network (**VBFLLFA**) analyzes the sequence of these metrics over time to predict whether the driver is **Alert** or **Drowsy**.

---

## 📊 The Dataset (YawDD)
This model is trained and evaluated on the **Yawning Detection Dataset (YawDD)**, which features real-world dashboard camera footage of drivers of varying demographics.

### Data Filtering & Preprocessing
To ensure high-quality training data, strict filtering protocols are applied during ingestion:
* **Demographic Isolation:** The pipeline strictly targets the `FemaleNoGlasses` and `MaleNoGlasses` subsets.
* **Session Threshold:** Subjects with fewer than 3 recorded sessions are dropped to guarantee sufficient data density per individual.

### Label Mapping & Binarization
The raw dataset contains multi-class behavioral labels. To frame this as a fatigue detection problem, the labels are mapped into a binary target variable:
* **Class 0 (Alert):** "Normal" driving and "Talking".
* **Class 1 (Drowsy):** "Yawning" and "Talking while Yawning".

---

## 📐 Mathematical Feature Engineering
Rather than feeding raw pixels into the neural network, the pipeline leverages the `dlib` 68-point facial landmark predictor to extract four continuous mathematical features per frame. 

1. **EAR (Eye Aspect Ratio):** Tracks the vertical vs. horizontal distance of the eyelids to detect blinks and microsleeps.
   $EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$

2. **MAR (Mouth Aspect Ratio):**
   Tracks the opening of the mouth to identify speaking or yawning.

3. **Head Tilt (Pose Angle):**
   Calculates the degree angle using the coordinates of the left and right eye centers, identifying if the driver's head is dropping.

4. **PERCLOS (Percentage of Eye Closure):**
   A rolling metric that calculates the percentage of recent frames where the driver's EAR dropped below a critical threshold (typically $0.25$).

### The Sliding Window (Temporal Context)
Because fatigue is an event that occurs over time, these 1D features are grouped using a **Sliding Window Algorithm**. 
* **Window Size:** `16 frames`
* **FPS:** `20 FPS`
* **Real-World Time:** Every sequence represents exactly `0.8 seconds` of continuous driver behavior.
* **Tensor Shape:** `[Batch_Size, 16, 4]`

---

## 🧠 VBFL Model Architecture
**VBFL** (*Video-Based Fatigue Learning*) is a custom PyTorch architecture built for speed and temporal reasoning.

### Layer-by-Layer Breakdown
1. **CSP Layer (Spatial Projection):** A trainable linear projection layer. It takes the `[16, 4]` input tensor and scales the 4 raw spatial features into a 32-dimensional embedding space (`[16, 32]`). This provides the necessary mathematical depth for the Transformer.
2. **Transformer Encoder (Temporal Attention):**
   The core engine. It utilizes Multi-Head Self-Attention (4 Heads, 2 Layers) to analyze the 16 frames simultaneously. It learns complex temporal relationships—for example, distinguishing the fast mouth movement of a shout from the slow, sustained mouth opening of a yawn.
3. **Global Average Pooling (GAP):**
   Collapses the temporal dimension (`[16, 32]` $\rightarrow$ `[32]`). It calculates the mathematical average across the 16 frames, creating a single, unified summary vector that prevents the model from overfitting to specific time steps.
4. **Classification Head (MLP):**
   A dense neural network pipeline: `LayerNorm` $\rightarrow$ `Linear(64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(1)`.
5. **Final Output:**
   A single raw logit. The model uses an implicit Sigmoid activation and a `0.5` threshold to assign the final `Alert` or `Drowsy` label.

---

## ⚙️ Pipeline Workflow

### 1. Data Ingestion
* Authenticates via the Kaggle API.
* Downloads and unzips the YawDD dataset dynamically.
* Parses directories using Regex to extract subject IDs and metadata.

### 2. Feature Extraction
* Normalizes all video files to 20 FPS using OpenCV.
* Iterates through every frame, applies the `dlib` landmark detector, calculates EAR/MAR/Tilt/PERCLOS, and exports the data to `temporal_features.csv`.

### 3. Cross-Validation Split
* Implements a **5-Fold Stratified Group K-Fold** methodology.
* **Leakage Prevention:** Data is grouped by `subject_id`. This guarantees that frames from a specific person are never present in both the training and validation sets simultaneously, ensuring the model generalizes to unseen faces.

### 4. Training Loop
* **Standardization:** Applies `StandardScaler` (Z-score normalization) strictly fitted on the training fold to prevent massive metrics (like Head Tilt) from drowning out small decimals (like EAR).
* Trains for a maximum number of epochs using the `Adam` optimizer and `ReduceLROnPlateau` learning rate scheduler.
* Utilizes **Early Stopping** (Patience = 8) to halt training if validation metrics stagnate.

### 5. Evaluation & Post-Processing
* Generates sequence-level predictions.
* **Majority Voting:** Groups the 16-frame sequence predictions by `Video_ID`. If the majority of sequences in a video are flagged as Class 1, the entire video is definitively labeled as **Drowsy**.
* Generates Classification Reports, Confusion Matrices, and ROC-AUC curves for final model analysis.

---

## ⚖️ Advanced Training Mechanics

### Solving Class Imbalance (`pos_weight`)
The dataset inherently contains more "Alert" frames than "Drowsy" frames. To prevent the model from artificially inflating its accuracy by always guessing "Alert," the pipeline modifies the loss function.
* Calculates the exact imbalance ratio (Negative Samples / Positive Samples).
* Injects this ratio as the `pos_weight` parameter into PyTorch's `BCEWithLogitsLoss`.
* **Result:** The model receives a multiplied mathematical penalty every time it misses a Drowsy frame, forcing it to maintain high sensitivity to fatigue markers.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* CUDA-capable GPU (Highly Recommended)
* CMake and a C++ Compiler (Required for compiling `dlib`)
* pip install opencv-python dlib pandas numpy scikit-learn matplotlib seaborn torch torchvision