from config import *
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import config

"""
Load data

Args:
    faces: matrix contains faces data
    n: number of rows
    m: number of columns
"""
mat_contents = scipy.io.loadmat(config.PATH) # Load data from file
faces = mat_contents['faces']
m = int(mat_contents['m'][0, 0]) # Number of row
n = int(mat_contents['n'][0, 0]) # Number of column
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

x = faces.T # Data transformation
y = np.zeros((faces.shape[1], 1)) # Create labels

j = 0
classes = list(range(len(nfaces)))
for i in nfaces:
    y[j:j+i] = classes.pop(0)
    j = j + i

"""
Split data into train and test

Args: 
    x: dataset labels
    x_train: training set features
    x_test: testing set features
    y: dataset labels
    y_train: training set labels
    y_test: testing set labels
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.SIZE, shuffle=True, stratify=y)

"""
Compute PCA

Args: 
    n_components: number of components to keep after the transformation
    svd_solver: algorithm to use for singular value decomposition  
"""
pca = PCA(n_components=config.N_COMPONENTS, svd_solver='randomized', whiten=True).fit(x_train)

eigenfaces = pca.components_.reshape((config.N_COMPONENTS, m, n))

x_train_pca = pca.transform(x_train) # Fits the PCA model to the training data
x_test_pca = pca.transform(x_test) # Extracting Eigenfaces

"""
Train a SVM classification model
"""
param_grid = {'C': config.C, 'gamma': config.GAMMA, } # Define grid
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid) # Initiate
clf = clf.fit(x_train_pca, y_train.ravel()) # Perform grid search

"""
Show classfification report
"""
#show_classification_report(x_test_pca, y_test, clf)

"""
Show confusion matrix
"""
#show_confusion_matrix(x_test_pca, y_test, clf)

"""
Show a portion of predictions
"""
#show_prediction(x_test_pca, x_test, y_test, m, n, clf)