import argparse  # The argparse module provides a powerful and user-friendly way to parse arguments passed to your script from the command line.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from data_preprocess import   load_and_process_data_emotion, load_and_process_data_emotion_new ,load_and_process_data_emotion_test
from sklearn.impute import KNNImputer #  imports the KNNImputer class from the impute submodule of the scikit-learn library.
import featureExtractor as fe
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Pandas is a powerful and popular library for data analysis and manipulation.
import seaborn as sns   #  to create statistical data visualizations.

def knnValueImputer(input_data):
    # create an object for KNNImputer
    imputer = KNNImputer(n_neighbors=1)  # To impute the missing value in given test data by the value of n_neighbors = 1
    output_data = imputer.fit_transform(input_data)
    return output_data



def plotconfusion(y_actu, y_pred ):
    from sklearn.metrics import confusion_matrix

    # cm = confusion matrix
    cm = confusion_matrix(y_actu, y_pred)

    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")

    #  def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):


    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                yticklabels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()


if __name__ == '__main__':
    # Create an argument for the script to select the dataset

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset', help='Choose one of the datasets:emotion', choices=[ 'emotion'], default='emotion')
    args = parser.parse_args()

    if args.dataset == 'emotion':

        # Number of (train, test) points for knn circuit and load data
        M, N = 128, 30
        # _, x_train, y_train, _, x_test, y_test, unique_map = load_and_process_data_emotion(first, second, M, N)
        _, x_train, y_train, _, x_test, y_test = load_and_process_data_emotion_new()
        x_train = knnValueImputer(x_train)

    # Build the classifier object
    classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1) # n_job = controls the number of CPU cores used during the fitting process of the KNN classifier.


    # Fit (train) the classifier on the training data
    classifier.fit(x_train, y_train)   #learns from your data and builds its internal representation for classification
    x_test = knnValueImputer(x_test)

    # Get the predictions from the test data of FER2013 dataset
    predictions = classifier.predict(x_test)

    # Count the number of correct predictions
    num_correct = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            num_correct += 1

    # Print the Test Accuracy
    # print(f'Data Map: {unique_map}')
    print(f'Accuracy: {100 * num_correct / len(y_test):.2f}%')

    plotconfusion(y_test, predictions)

    # Test KNN model on unknown data
    _, x_test_new, y_test_new = load_and_process_data_emotion_test()
    x_test_new = knnValueImputer(x_test_new)

    # Get the predictions from the test data of User Defined dataset
    predictions = classifier.predict(x_test_new)

    # Count the number of correct predictions
    num_correct = 0
    for i in range(len(y_test_new)):
        if predictions[i] == y_test_new[i]:
            num_correct += 1





    # Print the Test Accuracy
    # print(f'Data Map: {unique_map}')
    print(f'Accuracy on unknown data: {100 * num_correct / len(y_test_new):.2f}%')
    plotconfusion(y_test_new, predictions)
