# Cat vs Dog Image Classifier

A robust Convolutional Neural Network (CNN) solution for classifying images as cats or dogs, leveraging TensorFlow/Keras and modern deep learning best practices. This project covers the entire pipeline, from dataset preparation and cleaning to interactive model deployment via a web interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation & Requirements](#installation--requirements)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Web Demo](#web-demo)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Project Overview

This repository provides an end-to-end deep learning workflow for binary image classification (cat vs dog). It demonstrates data preprocessing, augmentation, model construction, evaluation, and deployment using Gradio for user-friendly real-time inference.

---

## Features

- **Automated Dataset Cleaning:** Ensures only valid images are used for training and validation.
- **Data Augmentation:** Enhances generalization through real-time transformations.
- **State-of-the-Art CNN:** Employs convolutional, batch normalization, and dropout layers for optimal performance and regularization.
- **Training Utilities:** Integrates early stopping and learning rate scheduling.
- **Result Visualization:** Offers training/validation accuracy plots for performance tracking.
- **Interactive Inference:** Supports image uploads for instant predictions.
- **Web Deployment:** Launches a Gradio-powered web interface for seamless sharing and testing.

---

## Dataset

- **Source:** [Kaggle Dogs vs. Cats Dataset (PetImages)](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
- **Preprocessing:** The notebook automatically verifies and selects 6,000 clean images per class (cat/dog), splitting them 80/20 for training and validation.

---

## Installation & Requirements

**Recommended Environment:** Google Colab (for effortless GPU access and Google Drive integration)

**Dependencies:**

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- matplotlib
- numpy
- Pillow
- Gradio


## Model Architecture

- **Input:** 128x128 RGB images
- **Convolutional Blocks:** 3× [Conv2D → BatchNorm → MaxPooling]
- **Dense Layers:** Flatten → Dense(128) → Dropout(0.5)
- **Output Layer:** Dense(1), sigmoid activation (binary output)
- **Regularization:** L2 penalties and Dropout applied

---

## Results

- **Final Training Accuracy:** ~87%
- **Final Validation Accuracy:** ~86%
- **Visualization:** Accuracy curves are plotted for both training and validation phases.

---

## Web Demo

After training, launch the Gradio interface in the notebook. A shareable link will be generated for real-time image classification.

---

## Acknowledgments

- [Microsoft Dogs vs. Cats Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
- [TensorFlow](https://www.tensorflow.org/)
- [Gradio](https://gradio.app/)

---

## License

This project is provided for educational and research purposes. Please check individual library and dataset licenses for their respective use cases.

---

*For questions, suggestions, or contributions, please open an issue or contact the repository maintainer.*
