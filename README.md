# Eigenfaces

This project implements Eigenfaces, a facial recognition technique based on Principal Component Analysis (PCA), in Python. Eigenfaces are commonly used for face recognition tasks.

## Overview

Eigenfaces is a dimensionality reduction technique that represents facial features as principal components obtained through PCA. The algorithm extracts features from facial images and uses them to identify and recognize faces.

## Project Structure

- `eigenfaces.py`: Contains the source code for the Eigenfaces implementation.
- `config.py`: Store configurations and utilities.
- `allFaces.mat`: Store sample datasets.
- `README.md`: Documentation and project overview.

## Dependencies

- Python 3.x
- NumPy
- scikit-learn
- Matplotlib
- SciPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nguyentuantai792003/eigenfaces.git
```

2. Install dependencies:

- Use 'pip install' to install required dependencies

3. Run the script:

- Remove the command sign # of the function you want to use:
    - show_classification_report(x_test_pca, y_test, clf): show classification report
    - show_confusion_matrix(x_test_pca, y_test, clf): show confusion matrix
    - show_prediction(x_test_pca, x_test, y_test, m, n, clf): show a portion of predictions

- Run python script

```bash
python3 eigenfaces.py
```
