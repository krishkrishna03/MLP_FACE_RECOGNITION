# MLP Face Recognition

## Overview

This project implements a **Multilayer Perceptron (MLP)** model for face recognition using Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). The goal is to build a system that can classify faces into different categories based on a dataset of labeled face images. The dataset is expected to have one folder for each individual, and each folder contains multiple images of that individual.

### Key Features
1. **PCA for Dimensionality Reduction:** Reduces the dimensionality of the data by extracting the principal components of the face images.
2. **LDA for Classification:** Uses LDA to project the PCA-transformed data into a space that maximizes the separability between different classes.
3. **MLP Classifier for Face Recognition:** An MLP classifier is used to learn the patterns of faces and predict the person’s identity based on the transformed features.

## Dataset Structure

The dataset should be structured as follows:

```
dataset/
    person1/
        img1.jpg
        img2.jpg
        ...
    person2/
        img1.jpg
        img2.jpg
        ...
    ...
```

- Each folder (e.g., `person1`, `person2`, etc.) represents an individual.
- Each image in a folder corresponds to an image of that individual.

### Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scipy`
  - `opencv-python`
  - `scikit-learn`
  - `keras`
  - `tensorflow`
  - `matplotlib`

You can install the necessary libraries using the following command:

```bash
pip install numpy scipy opencv-python scikit-learn keras tensorflow matplotlib
```

### Code Explanation

1. **Image Preprocessing:**
   - The images are read and converted to grayscale using OpenCV.
   - Each image is resized to a standard dimension (300x300).
   - All images are then flattened into vectors, and stored in an array `X`, while the corresponding labels are stored in `y`.

2. **PCA (Principal Component Analysis):**
   - PCA is performed on the training set to reduce the dimensionality of the data and extract the most important features (eigenfaces).
   - The eigenfaces are plotted for visualization purposes.

3. **LDA (Linear Discriminant Analysis):**
   - LDA is applied on the PCA-transformed data to enhance the class separability.

4. **MLP Classifier:**
   - An MLP classifier is trained using the LDA-transformed data.
   - The classifier is used to predict the labels for the test set, and accuracy is calculated.

5. **Model Evaluation:**
   - The accuracy of the model is calculated by comparing the predicted and actual labels.
   - The faces that are correctly classified are displayed along with their predicted labels and probabilities.

### Key Functions

- **`plot_gallery(images, titles, h, w, n_row=3, n_col=4)`**:
  - Visualizes the images (e.g., eigenfaces, test faces) in a grid format.

- **`train_test_split(X, y)`**:
  - Splits the dataset into training and testing sets.

- **PCA and LDA transformation**:
  - Reduces the dataset dimensions and projects the data into a new space.

- **MLP Classifier training and evaluation**:
  - Trains the MLP classifier and evaluates its performance using the test data.

### Sample Run

**Input:**
1. A dataset with folders representing individuals and their images.
2. The program preprocesses the images and performs PCA and LDA to extract relevant features.

**Output:**
- Eigenfaces visualization.
- Model accuracy and predictions.
- A gallery of test images with predicted labels and probabilities.

### Results

After running the code, the system will output the accuracy of the face recognition system along with a gallery of predicted vs. true faces. For instance:

```bash
Accuracy: 85.6%
```

### Conclusion

This project provides a basic approach to face recognition using machine learning techniques such as PCA for dimensionality reduction, LDA for improving class separability, and an MLP classifier for the final prediction. This approach can be extended and improved by using more advanced models and techniques, such as deep learning-based architectures.

### Future Work

- Experiment with more sophisticated models such as Convolutional Neural Networks (CNNs) for improved accuracy.
- Implement real-time face recognition using webcam input.
- Handle larger datasets with more efficient training techniques.

---

This is a basic but functional implementation of face recognition using a neural network-based approach, ideal for learning and experimentation in the field of computer vision.
