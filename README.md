# Computer-Vision-project-CIFAR-10

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Networks-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/PyTorch-Transfer%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/JAX-Flax-000000?logo=google&logoColor=white" alt="JAX Flax">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

## 1. Project Overview
This project explores and compares multiple convolutional neural network (CNN) architectures for image classification using the CIFAR-10 dataset.

The main objectives were:

- Build a Convolutional Neural Network (CNN) from scratch
- Implement transfer learning using pretrained models
- Apply data augmentation and regularization techniques
- Compare models using evaluation metrics
- Analyze performance using confusion matrices

The project was developed in VS Code with the extension Colab and results were stored in Google Drive for reproducibility.

---

## 2. Dataset Description

[CIFAR-10](https://keras.io/api/datasets/cifar10/)

- 60,000 images
- 10 classes
- 32×32 RBG images
- 6,000 images in 10 classes

Images were resized to 96×96 to improve compatibility with pretrained ImageNet models while keeping computational cost manageable.

### Data was divided into:

- 40,000 training
- 10,000 validation
- 10,000 test

This division was made to avoid data leakage.

---

## 3. Methodology & Steps

#### 1. Data processing
    - Load the data
    - Split the data
    - Normalization applied to the images pixels
    - One-Hot Encoding applied to the labels

#### 2. Create a CNN base model (model 2)
    - Convolutional layers
    - ReLU activation
    - Dense classifier
    - Softmax output
    - Batch Normalization
    - Dropout
    - Early Stopping
    - ReduceLROnPlateau

#### 3. Tune CNN base model using VGG-BN model (model 3)
    - Increased depth
    - Data augmentation (with cutout) and generators
    - Global Average Pooling
    - Improved regularization
    - Optimized architecture
    - Loss with label smoothing + AdamW + CosineDecay

#### 4. Transfer Learning Models
    -  Pretrained ImageNet architectures:
      - EfficientNetB0 (model 04)
      - EfficientNetV2B1 (model 06)
      - EfficientNetV2S (PyTorch version) (model 07)

    - Transfer Learning Adjustments:
      - Images resized to 96×96
      - Data augmentation applied:
        - Rotation
        - Flip
        - Brightness adjustment
        - Contrast adjustment
        - Random cropping
      - Preprocessing applied according to each architecture
      - Base model initially frozen
      - Fine-tuning performed by unfreezing half of the layers
      - Smaller learning rates used during fine-tuning
      - Label smoothing applied in some experiments

    - Training Strategies:
      - Adam optimizer
      - Early Stopping
      - ReduceLROnPlateau
      - Learning rate scheduling
      - Label smoothing

#### 5. Model CNN with JAX/Flax (model 5)
  
    JAX:
    - Augmentation
      - Flip image (left or right)
      - Brightness
      - Contrast
      - Random Cutout
    - Batching
    
    Flax:
    - Convolutional layers
    - GeLU activation
    - Dense classifier
    - Softmax output
    - Batch Normalization
    - MaxPooling
    - Dropout
    - Early Stopping
    - ReduceLROnPlateau
    - Global avarage pooling (GAP)

#### 6. Stucking Model:
    - Stucked de models:
      - EfficientNetB0 (model 04)
      - EfficientNetV2B1 (model 06)
      - EfficientNetV2S (PyTorch version) (model 07)

This step was create  to try to increse the models accuracy.

---

## 4. Repository Policy

- `data/`, `models/`, and `backup/` are intentionally excluded from Git tracking.
- Datasets, logs, and trained weights are kept local.
- Final shareable outputs should go to `reports/` and visual figures to `images/`.

---

#### Common local outputs:

- Local experiment logs and CSV tracking: `data/logs/`, `data/model_performance_report.csv`, `data/experiments.csv`
- Local trained models: `models/`
- Figures and plots: `images/`

---

## 5. Repository Structure

```text
cifar10_project/
├── data/                      # Local only (not tracked)
│   ├── processed/
│   │   └── cifar10_processed.npz
│   ├── logs/
│   ├── model_performance_report.csv
│   └── experiments.csv
├── models/                    # Local only (not tracked)
├── notebooks/
│   ├── 01_data_processing.ipynb
│   ├── 02_cnn_model.ipynb
│   ├── 03_cnn_model_tuned.ipynb
│   ├── 04_transfer_learning_EfficientNetB0.ipynb
│   ├── 05_cnn_model_jax_flax.ipynb
│   ├── 06_transfer_learning_EfficientNetV2B1.ipynb
│   ├── 07_transfer_learning_EfficientNetV2S_PyTorch.ipynb
│   ├── 08_francesinha_meta_training.ipynb
│   └── 09_francesinha_inference_pretrained_meta.ipynb
├── images/                    # Figures (can be shared)
├── reports/                   # Report-ready outputs (can be shared)
├── utils/
│   ├── __init__.py
│   └── ml_utils.py
├── requirements-base.txt
├── requirements-tf.txt
├── requirements-torch.txt
├── requirements-jax.txt
└── README.md
```

---

## 6. How to Reproduce the Project (Optional – Draft)

#### Prerequisites
- Python 3.x
- TensorFlow / Keras
- PyTorch, JAX/Flax
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

#### Running the Project
Install requirements
Run notebooks in /notebooks
Models saved in /models
Performance reports generated automatically

### Run order
1. `notebooks/01_data_processing.ipynb`
2. `notebooks/02_cnn_model.ipynb`
3. `notebooks/03_cnn_model_tuned.ipynb`
4. `notebooks/04_transfer_learning.ipynb`
5. `notebooks/05_cnn_model_jax_flax.ipynb`
6. `notebooks/06_transfer_learning_EfficientNetV2B1.ipynb`
7. `notebooks/07_transfer_learning_EfficientNetV2S_PyTorch.ipynb`
8. stacked model - notebook to be added

---

## 7.Reuse in other projects

- Option 1 (simple): copy the `utils/` folder into the new project.
- Option 2 (recommended): package it as a small internal library and install with pip in your other projects.

The `utils/` folder contains model-agnostic helpers for:

- Image normalization (`normalize_images`, `normalize_splits`)
- Label handling (`ensure_label_indices`)
- Augmentation (`random_cutout`)
- TensorFlow dataset pipelines (`build_tf_dataset`)
- Metric extraction and experiment logging (`extract_final_history_metrics`, `compute_classification_metrics`, `evaluate_and_log_model`, `log_metrics_to_csv`)
- Log folder creation (`make_run_log_dir`)

---

## 8. Reports

The `reports/` folder is intended for final, shareable project outputs such as:

- Presentation exports (`.pdf`, `.pptx`)
- Final summary tables (`.csv`, `.md`)
- Consolidated evaluation documents

---

## 9. Authors

Ironhack Data Science & Machine Learning Bootcamp project.

This project was developed by:

- **Janete Barbosa**
  - GitHub: https://github.com/janeteccbarbosa28-eng
  - LinkedIn: https://linkedin.com/in/janete-barbosa
- **Alexandre Andrade**
  - GitHub: https://github.com/alexandrade1978
  - LinkedIn: https://www.linkedin.com/in/alexandre-andrade-12908b394
- **Isis Hassan**
  - GitHub: https://github.com/isishassan
  - LinkedIn: https://linkedin.com/in/isishassan


---

## 10. License

This project uses the MIT License
