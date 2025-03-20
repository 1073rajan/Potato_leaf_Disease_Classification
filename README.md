# Deep Learning Model for Potato Leaf Disease Classification

## Overview
This project implements a deep learning model using TensorFlow and Keras to classify potato leaf images into three categories:  
- `Potato___Early_blight`  
- `Potato___Late_blight`  
- `Potato___healthy`  

The model is trained on the [PlantVillage dataset](https://plantvillage.psu.edu/) with `256x256` RGB images.

---

## Dataset
- **Total Images**: 2,152  
- **Classes**: 3 (Early Blight, Late Blight, Healthy)  
- **Image Shape**: `(256, 256, 3)`  
- **Train-Validation-Test Split**:  
  - Training: 80%  
  - Validation: 10%  
  - Testing: 10%  

---

## Model Architecture
A Convolutional Neural Network (CNN) with the following layers:

### Data Preprocessing
- Image resizing and rescaling (`1/255.0` normalization)
- Data augmentation:
  - Random horizontal/vertical flip
  - Random rotation

### Convolutional Layers
1. `Conv2D` (32 filters, kernel size=3, activation='ReLU') → `MaxPooling2D`
2. `Conv2D` (64 filters, kernel size=3, activation='ReLU') → `MaxPooling2D`
3. `Conv2D` (128 filters, kernel size=3, activation='ReLU') → `MaxPooling2D`

### Fully Connected Layers
1. `Flatten`
2. `Dense` (128 units, activation='ReLU')
3. `Dense` (3 units, activation='Softmax')  

**Optimizer**: Adam  
**Loss**: Sparse Categorical Crossentropy  

---

## Training Details
- **Epochs**: 50  
- **Batch Size**: 32  
- **Metrics**: Accuracy  

### Performance
- **Final Training Accuracy**: ~99%  
- **Final Validation Accuracy**: ~100%  
- **Test Accuracy**: ~98%  

---

## Results Visualization

### Length of Reviews
The distribution of the Training and Validation Accuracy Curve.

![Training and Validation Accuracy](Length_of_review.png)

The distribution of the Training and Validation Loss Curve.

![Training and Validation Loss](Length_of_review.png)

**Training accuracy stabilizes at** ~99%.

**Validation accuracy peaks at** 100%.

**Loss decreases consistently across epochs**.

## Installation & Usage

### Prerequisites
```bash
pip install tensorflow numpy matplotlib
