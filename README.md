# EEG Subject Identification using CNN–Transformer Models

## 📌 Project Overview
This project explores large-scale EEG-based subject identification using deep learning, leveraging the PhysioNet EEG Motor Movement/Imagery dataset containing recordings from 109 subjects. EEG signals are inherently high-dimensional, noisy, and highly variable across individuals, making subject-level classification a challenging yet meaningful problem for neuroscience, biometric systems, and Brain–Computer Interface (BCI) research.

The dataset consists of 64-channel EEG recordings stored in EDF+ format, captured during motor execution and motor imagery tasks under a standardized experimental protocol. Each recording is preprocessed using bandpass filtering, fixed-length epoching, and temporal resampling to produce uniform EEG segments of shape (channels × time). These segments preserve both spatial electrode relationships and temporal neural dynamics, forming the foundation for deep neural modeling.

To effectively capture multi-scale EEG patterns, this work implements a CNN + Bidirectional RNN (BiLSTM) architecture. The convolutional layers learn local temporal features and channel-wise representations, while stacked bidirectional LSTM layers model long-range temporal dependencies across EEG sequences. This hybrid design enables the network to extract subject-specific neural signatures that remain stable across trials and sessions.

The model is trained end-to-end using cross-entropy loss and evaluated on a held-out test set using accuracy, precision, recall, and F1-score. Additional analyses include per-subject accuracy breakdowns, confusion matrix visualizations, and learning curve inspection to assess generalization and class imbalance effects. The results demonstrate that deep spatio-temporal modeling can successfully differentiate individuals based solely on EEG activity, even in a high-class (109-subject) setting.

Overall, this project highlights the feasibility of EEG-based biometric identification using deep learning and provides a scalable framework for future research in subject recognition, personalized BCI systems, and neuroadaptive technologies.

---

## 📂 EEG Data Format (EDF Background)

EEG recordings are provided in EDF (European Data Format), a standard format for biomedical time-series data.

Each EDF file contains:

### 🔹 Header Information
- Subject identifier
- Recording metadata
- Number of EEG channels (e.g., 64 electrodes)
- Sampling frequency (Hz)
- Recording duration
- Channel names (Fp1, Fp2, C3, C4, etc.)

### 🔹 Signal Data
- Continuous EEG voltage values (in microvolts)
- Synchronized multi-channel time-series signals
- Sampled at a fixed frequency (e.g., 128 Hz / 160 Hz)

### 🔹 Annotations (if available)
- Event markers or task labels
- Used to guide signal segmentation

---

## 🔄 Data Preprocessing Pipeline

1. **EDF Loading**
   - EDF files are read using EEG processing libraries
   - Raw EEG signals are extracted for all channels

2. **Signal Segmentation (Epoching)**
   - Continuous EEG recordings are split into fixed-length windows
   - Each window represents one training sample
   - Example:
     - 2-second window
     - 256 time samples
     - 64 EEG channels

3. **Tensor Construction**
   - Model input tensor shape:
     ```
     (samples, channels, time_steps)
     (N, 64, 256)
     ```

4. **Label Encoding**
   - Each EEG segment is labeled with its corresponding subject ID
   - Subject IDs are mapped to integer class labels

5. **Train–Test Split**
   - Data is divided into training and testing sets
   - Example:
     ```
     Training data: (32576, 64, 256)
     Testing data:  (13961, 64, 256)
     ```

---

## 🧠 Dataset Description
- Multichannel EEG time-series signals
- 64 electrodes per sample
- Fixed-length temporal windows
- Labels correspond to subject identities
- Multi-class classification problem with many subjects

---

## 🏗️ Model Architecture

### 🔹 CNN + Transformer Hybrid Model

- **Convolutional Neural Network (CNN)**
  - Extracts local spatial and temporal features
  - Learns inter-channel and short-term temporal patterns

- **Transformer Encoder**
  - Uses self-attention to model long-range temporal dependencies
  - Captures global EEG dynamics within each segment

- **Fully Connected Layers**
  - Map learned representations to subject class probabilities

This hybrid architecture combines local feature extraction with global temporal modeling, making it well-suited for EEG-based identification tasks.

---

## ⚙️ Training Setup
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam / AdamW
- Learning Rate Scheduling: ReduceLROnPlateau
- Mini-batch training
- Gradient clipping for stability
- Model checkpointing based on validation accuracy

---

## 📈 Evaluation Metrics
- Overall classification accuracy
- Per-subject accuracy
- Confusion matrix analysis
- Training and validation loss curves
- Training and validation accuracy curves

---

## 📊 Confusion Matrix Analysis
- Full confusion matrix across all subjects
- Per-class accuracy distribution
- Sample distribution per subject
- High-resolution visualizations saved for analysis

---
