import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from tqdm import tqdm
from skimage import feature
from sklearn.svm import LinearSVC
from imutils import paths
import argparse


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist


labels = []
data = []

labelDict = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}


def getFeatureExtractedImageData():
    path = r"C:\Users\snehal\PycharmProjects\qlearnkit-master\dataset\images\train"
    desc = LocalBinaryPatterns(24, 8)
    os.chdir(path)
    for imagePath in paths.list_images(os.getcwd()):
        # print(imagePath)
        imagePath = imagePath.split("\\")[-2:]
        path = os.path.join(os.path.join(os.getcwd(), imagePath[0]) + "\\", imagePath[1])
        image = cv2.imread(path)
        # print(path)
        # plt.imshow(plt.imread(path))
        # plt.show()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        labels.append(labelDict[imagePath[-2]])
        # print(labels)
        # hist = [int(x) for x in hist]
        # norm = np.linalg.norm(hist)
        # hist = hist / norm
        data.append(hist)
    return np.array(data), np.array(labels)


testData = []
testLabels = []


def getTestingData():
    path = r"C:\Users\snehal\PycharmProjects\qlearnkit-master\dataset\images\validation"
    desc = LocalBinaryPatterns(24, 8)
    os.chdir(path)
    for imagePath in paths.list_images(os.getcwd()):
        imagePath = imagePath.split("\\")[-2:]
        path = os.path.join(os.path.join(os.getcwd(), imagePath[0]) + "\\", imagePath[1])
        image = cv2.imread(path)
        image = cv2.resize(image, (780, 540), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        # hist = [int(x) for x in hist]
        # norm = np.linalg.norm(hist)
        # hist = hist / norm
        testData.append(hist)
        testLabels.append(labelDict[imagePath[-2]])
        # print(labels)
        # data.append(hist)
        # prediction = model.predict(hist.reshape(1, -1))
        # cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # cv2.imshow("Image", image)
    return np.array(testData), np.array(testLabels)


def plotconfusion(y_actu, y_pred ):
    from sklearn.metrics import confusion_matrix

    # y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    # y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]

    confusion_matrix(y_actu, y_pred)

    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")

    #  def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

    # df_confusion = pd.crosstab(y_actu, y_pred)
    # plot_confusion_matrix(df_confusion)
# plotconfusion([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2],[0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2])
'''
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)
for imagePath in tqdm(paths.list_images(os.getcwd())):
    imagePath = imagePath.split("\\")[-2:]
    path = os.path.join(os.path.join(os.getcwd(), imagePath[0]) + "\\", imagePath[1])
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
'''
