import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

"""
Config value
"""
N_COMPONENTS = 1000
PATH = 'allFaces.mat'
SIZE = 0.4
C = [1e3, 5e3, 1e4, 5e4, 1e5]
GAMMA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

"""
Utils:
    plot_eigenfaces: generates a visual representation of the first five eigenfaces obtained from a PCA
    show_classification_report: display the classification report
    show_confusion_matrix: display the confusion matrix
    plot_gallery: visualize a collection of images in a grid
    show_prediction: visualize the predictions made by a classifier on test data
"""


def plot_eigenfaces(eigenfaces):
    fig1 = plt.figure(figsize=(10, 3))
    for i in range(5):
        ax = fig1.add_subplot(1, 5, i+1)
        img = ax.imshow(eigenfaces[i].T)
        img.set_cmap('gray')
        ax.set_title(f"Eigenface{i+1}")
        plt.axis('off')
    plt.show()

def show_classification_report(x_test_pca, y_test, clf):
    y_pred = clf.predict(x_test_pca)
    target_names = [f"Person{n}" for n in range(1,39)]
    print(classification_report(y_test, y_pred, target_names=target_names))

def show_confusion_matrix(x_test_pca, y_test, clf):
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = [10, 10]
    y_pred = clf.predict(x_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1,39))
    disp.from_estimator(clf, x_test_pca, y_test, cmap=plt.cm.Blues, display_labels=range(1,39))
    plt.show()

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[int(y_pred[i].item())]
    true_name = target_names[int(y_test[i].item())]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def plot_gallery(images, titles, h, w, n_row=5, n_col=6, rgb=False):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        if rgb == False:
          plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        else:
          plt.imshow(images[i])
        plt.title(titles[i], size=12)
        plt.axis('off')
    plt.show()

def show_prediction(x_test_pca, x_test, y_test, m, n, clf):
    y_pred = clf.predict(x_test_pca)
    target_names = [f"Person{n}" for n in range(1,39)]
    prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
    plot_gallery(x_test, prediction_titles, m, n)