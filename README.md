# EEG Subject Identification using CNN–Transformer Models

## 📌 Project Overview
This project explores large-scale EEG-based subject identification using deep learning, leveraging the PhysioNet EEG Motor Movement/Imagery dataset containing recordings from 109 subjects. EEG signals are inherently high-dimensional, noisy, and highly variable across individuals, making subject-level classification a challenging yet meaningful problem for neuroscience, biometric systems, and Brain–Computer Interface (BCI) research.

The dataset consists of 64-channel EEG recordings stored in EDF+ format, captured during motor execution and motor imagery tasks under a standardized experimental protocol. Each recording is preprocessed using bandpass filtering, fixed-length epoching, and temporal resampling to produce uniform EEG segments of shape (channels × time). These segments preserve both spatial electrode relationships and temporal neural dynamics, forming the foundation for deep neural modeling.

To effectively capture multi-scale EEG patterns, this work implements a CNN + Bidirectional RNN (BiLSTM) architecture. The convolutional layers learn local temporal features and channel-wise representations, while stacked bidirectional LSTM layers model long-range temporal dependencies across EEG sequences. This hybrid design enables the network to extract subject-specific neural signatures that remain stable across trials and sessions.

The model is trained end-to-end using cross-entropy loss and evaluated on a held-out test set using accuracy, precision, recall, and F1-score. Additional analyses include per-subject accuracy breakdowns, confusion matrix visualizations, and learning curve inspection to assess generalization and class imbalance effects. The results demonstrate that deep spatio-temporal modeling can successfully differentiate individuals based solely on EEG activity, even in a high-class (109-subject) setting.

Overall, this project highlights the feasibility of EEG-based biometric identification using deep learning and provides a scalable framework for future research in subject recognition, personalized BCI systems, and neuroadaptive technologies.

---

## 📊 About the Dataset

### 🔎 Context
This dataset was created by the developers of the **BCI2000 instrumentation system** and is one of the most widely used benchmark datasets in **Brain–Computer Interface (BCI)** research. It contains EEG recordings from **109 subjects** performing motor execution and motor imagery tasks.

The original dataset is hosted on **PhysioNet**. This version is provided on **Kaggle** to enable easy experimentation, reproducibility, and deep learning workflows.

🔗 **Dataset Link (Kaggle):**  
https://www.kaggle.com/datasets/gamalasran/physionet-eeg-motor-movement-imagery/data

---

### 📦 Dataset Summary
- **Subjects:** 109 healthy volunteers  
- **EEG Channels:** 64 channels (International 10–10 system)  
- **Sampling Rate:** 160 Hz  
- **File Format:** EDF+ (European Data Format Plus)  
- **Signal Type:** Continuous multichannel EEG time-series  

---

### 🧠 EEG Channel Configuration
- 64-channel EEG montage
- Electrodes placed according to the **International 10–10 system**
- Signals recorded as voltage values (µV) over time

---

### 🧪 Experimental Protocol (Runs)
Each subject performed **14 experimental runs**, where each run corresponds to a specific task. Correct interpretation of the data requires mapping the **run number** to the task type.

| Run # | Task Description |
|-----|------------------|
| 1 | Baseline (Eyes Open) |
| 2 | Baseline (Eyes Closed) |
| 3, 7, 11 | Motor Execution: Left vs Right Fist |
| 4, 8, 12 | Motor Imagery: Left vs Right Fist |
| 5, 9, 13 | Motor Execution: Both Fists vs Both Feet |
| 6, 10, 14 | Motor Imagery: Both Fists vs Both Feet |

---

### 🏷️ Understanding Annotations (Events)
EDF+ files include **event annotations** indicating when a task begins.

- **T0** → Rest / Baseline  
- **T1 & T2** → Task-specific actions (meaning depends on run type)

#### Event Code Interpretation
| Run Type | T1 Meaning | T2 Meaning |
|--------|-----------|-----------|
| 3, 7, 11 | Motion of Left Fist | Motion of Right Fist |
| 4, 8, 12 | Imagine Left Fist | Imagine Right Fist |
| 5, 9, 13 | Motion of Both Fists | Motion of Both Feet |
| 6, 10, 14 | Imagine Both Fists | Imagine Both Feet |

These annotations are used to segment continuous EEG signals into task-specific epochs.

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



## 🏗️ Model Architecture

### 🔹 CNN + BiLSTM EEG Classification Network

This model is a **CNN–RNN hybrid architecture** designed for large-scale EEG subject classification. It combines convolutional feature extraction with bidirectional temporal modeling to capture both local EEG patterns and long-range dependencies.

---

### 🔹 Input Representation

- EEG segments are formatted as **(batch_size, channels, time_steps)**
- Channels correspond to EEG electrodes; time_steps represent temporal samples.

---

### 🔹 Convolutional Feature Extractor (CNN)

The CNN backbone consists of **four stacked 1D convolutional blocks**, each including:
- 1D Convolution (kernel size = 3, padding = 1)
- Batch Normalization
- ReLU activation
- Max Pooling (stride = 2)

| Block | Output Channels |
|------|-----------------|
| 1 | 64 |
| 2 | 128 |
| 3 | 256 |
| 4 | 512 |

These layers learn hierarchical temporal features while progressively reducing temporal resolution.  
A **Dropout layer (p = 0.5)** follows the CNN stack for regularization.

---

### 🔹 Temporal Modeling with BiLSTM

CNN features are reshaped to **(batch_size, reduced_time_steps, 512)** and passed through two stacked **Bidirectional LSTM layers**:

- BiLSTM 1: hidden size 256  
- BiLSTM 2: hidden size 128  

Bidirectional recurrence captures both past and future context in EEG signals.  
**Dropout (p = 0.3)** is applied between LSTM layers.

---

### 🔹 Classification Head

The final forward and backward LSTM states are concatenated and fed into fully connected layers:

- 256 → 512 → 256 → *num_classes*

The output layer produces logits for **109 subject classes**, optimized using cross-entropy loss.

---

## 🧪 Training Setup

This project uses a structured training pipeline for EEG-based subject classification.

---

### 🔹 Data Preparation
- **Input:** EDF+ EEG files (64 channels)
- **Preprocessing:**
  - Bandpass filter: **1–40 Hz**
  - Fixed-length epoching: **2 seconds**
  - Resampled to **256 time points**
- **Final shape:** `(epochs, channels, 256)`

---

### 🔹 Labels & Splitting
- Each subject mapped to a unique class ID (**109 classes**)
- Custom random split:
  - **70% training**
  - **30% testing**

---

### 🔹 Normalization
- Channel-wise **z-score normalization**
- Prevents scale imbalance across electrodes

---

### 🔹 Training Input
- Converted to **PyTorch tensors**
- Input shape: `(batch_size, channels, time_steps)`
- Loaded using **DataLoader** with shuffling and pinned memory

---


## 📊 Results

This section presents a comprehensive evaluation of the proposed **CNN + Transformer** architecture for **EEG-based subject identification**. The model is evaluated on a **109-subject** classification task using standard performance metrics, confusion matrix analysis, subject-wise accuracy assessment, and training–testing convergence behavior.

---

### 📈 Model Performance Metrics

The final trained model achieves **high and well-balanced performance** across all evaluation metrics, demonstrating strong generalization capability for large-scale subject identification from EEG segments.

- **Task:** Subject Identification from EEG Segments  
- **Architecture:** CNN + Transformer  
- **Number of Subjects:** 109  

**Final Evaluation Metrics:**
- **Accuracy:** 0.9825  
- **Precision:** 0.9830  
- **Recall:** 0.9825  
- **F1-score:** 0.9825  

The close agreement between precision, recall, and F1-score indicates that the model maintains a robust balance between false positives and false negatives, even under a high-class-count setting.

<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/b7d6f627-dea5-4a27-ab86-7967b6de42c4" />


---

### 🧩 Confusion Matrix Analysis

The confusion matrix reveals strong diagonal dominance, indicating that EEG segments are correctly classified for the majority of subjects. Misclassifications are sparse and primarily limited to a small subset of subjects with overlapping EEG characteristics.

**Confusion Matrix Statistics:**
- **Total classifications:** 13,961  
- **Total correct classifications:** 13,717  
- **Mean accuracy per class:** 0.9825  
- **Highest individual class accuracy:** 1.0000  
- **Lowest individual class accuracy:** 0.9040  

These results confirm that the model performs consistently across subjects, with no severe class-level degradation.

Full confusion matrix for all 109 subjects  
<img width="1283" height="1189" alt="image" src="https://github.com/user-attachments/assets/627eb02a-b592-4e4c-8401-d8911819be56" />


Zoomed-in confusion matrix for selected subjects  
<img width="1104" height="989" alt="image" src="https://github.com/user-attachments/assets/58b16407-7e5b-4479-a1a1-a6d68699c084" />


---

### 👥 Subject-wise Accuracy Analysis

To further analyze class-level behavior, subject-wise classification accuracy was computed from the confusion matrix diagonal. The results show that most subjects achieve near-perfect identification accuracy.

- **Mean subject-wise accuracy:** 0.9825  
- **Accuracy range across subjects:** 0.9040 – 1.0000  

Only a small number of subjects exhibit slightly reduced accuracy, likely due to higher intra-subject EEG variability or inter-subject similarity.

<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/709a10c4-a4be-42ce-abb2-27dc74b24a2f" />


---

### 📊 Samples per Subject Distribution

The number of EEG samples per subject was analyzed to ensure that classification performance is not biased by data imbalance.

- Sample counts are reasonably distributed across subjects  
- Mean sample count aligns with stable subject-wise performance  

This indicates that the observed classification accuracy is not driven by over-representation of specific subjects.

<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/e6a1eaef-7ddc-43d9-b6e0-25ebe85c96af" />


---

### 📉 Training and Test Loss Analysis

The training and test loss curves demonstrate **stable convergence behavior** across epochs.

- Training loss decreases steadily  
- Test loss closely follows training loss  
- No significant divergence is observed  

This confirms that the model effectively learns discriminative EEG features without overfitting.

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/1101463b-3679-4003-aa7d-4ef64083c9e4" />


---

### 📈 Training and Test Accuracy Analysis

Accuracy trends further validate the stability of the training process.

- Training and test accuracy curves increase consistently  
- Minimal performance gap between training and test accuracy  

This behavior indicates strong generalization to unseen EEG segments.

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/363dc263-41b4-44f4-b144-9e260942137a" />



---

### ✅ Overall Results Summary

The experimental results demonstrate that the proposed **CNN + Transformer** model effectively captures subject-specific EEG patterns in a challenging **109-class identification problem**. High overall accuracy, consistent subject-wise performance, strong confusion matrix structure, and stable training dynamics collectively validate the robustness of the proposed approach for EEG-based biometric identification and neurosecurity applications.
