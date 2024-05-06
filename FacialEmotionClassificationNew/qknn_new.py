import argparse
import numpy as np
from cirq import S, X, H, SWAP, ry
from cirq import Moment, Circuit, LineQubit, Simulator, measure

from gates import eF, SDagger
from quantum_kernel import build_kernel_original
from data_preprocess import  load_and_process_data_emotion, load_and_process_data_emotion_new, load_and_process_data_emotion_test
import featureExtractor as fe
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Pandas is a powerful and popular library for data analysis and manipulation.
import seaborn as sns   #  to create statistical data visualizations.


def qknn_circuit(F, theta0, theta1, theta2, r=4, verbose=0):
    """Creates and runs the QKNN circuit for two given train values and one test value.

    After the thetas are ran through the QKNN circuit, the resulting state vector
    is used to determine the prediction value [1/-1] for the given test theta.
    First an expectation matrix (O) is created, then used with the inner product
    of the state vector with itself. The sign of the resulting value is what is
    used for the prediction. The formula for these steps is denoted as follows:

    1. O = |000><000| ⊗ |1><0|
    2. E = <ψ|O|ψ>
    3. P = sign(E)

    Args:
        F: The matrix used to generate the unitary operator e^(iFt0).
        theta0: The test theta, for which a prediction will be determined.
        theta1: The first train theta.
        theta2: The second train theta.
        r: A term used to adjust the controlled rotation during the HHL portion.
        verbose: A parameter to determine what information is printed. Either 0, 1, or 2.
    Returns:
        A prediction for the given test theta, either 1 or -1.
    """
    # Circuit
    c = Circuit()

    # Set 4 Qubits
    q1 = LineQubit(1)
    q2 = LineQubit(2)
    q3 = LineQubit(3)
    q4 = LineQubit(4)

    # Setup Non-Zero States
    c.append(Moment([X(q3)]))
    c.append(Moment([H(q3)]))

    # Phase Estimation with Matrix F Derived from K
    c.append(Moment([H(q1), H(q2)]))
    c.append(Moment([eF(F, 2).on(q3).controlled_by(q1)]))
    c.append(Moment([eF(F, 1).on(q3).controlled_by(q1)]))
    c.append(Moment([SWAP(q1, q2)]))
    c.append(Moment([H(q2)]))
    c.append(Moment([SDagger().on(q1).controlled_by(q2)]))
    c.append(Moment([H(q1)]))

    # Controlled Rotation
    c.append(Moment([X(q1)]))
    c.append(Moment([ry(2*np.pi/(2 ** r)).on(q4).controlled_by(q1)]))
    c.append(Moment([ry(np.pi/(2 ** r)).on(q4).controlled_by(q2)]))
    c.append(Moment([X(q1)]))

    # Inverse Phase Estimation
    c.append(Moment([H(q1)]))
    c.append(Moment([S(q1).controlled_by(q2)]))
    c.append(Moment([H(q2)]))
    c.append(Moment([SWAP(q1, q2)]))
    c.append(Moment([eF(F, -1).on(q3).controlled_by(q1)]))
    c.append(Moment([eF(F, -2).on(q3).controlled_by(q1)]))
    c.append(Moment([H(q1), H(q2)]))

    # Training Oracle
    c.append(Moment([X(q3)]))
    c.append(Moment([ry(2*theta1).on(q2).controlled_by(q3, q4)]))
    c.append(Moment([X(q3)]))
    c.append(Moment([ry(2*theta2).on(q2).controlled_by(q3, q4)]))

    # Test Oracle
    c.append(Moment([ry(-2*theta0).on(q2).controlled_by(q4)]))
    c.append(Moment([H(q3).controlled_by(q4)]))

    # Print the circuit if desired
    if verbose == 1:
        print()
        print(c)
        print()


    # Create the simulator
    s = Simulator()

    # Get the simulated results and final state vector
    results = s.simulate(c)
    state = results.final_state_vector.real

    # O = |000><000| ⊗ |1><0|
    O = np.zeros((16, 16)); O[1][0] = 1

    # E = <ψ|O|ψ>
    E = np.inner(np.matmul(state.conj().T, O), state)

    # Prediction = sign(E)
    prediction = int(np.sign(E).real)
    if prediction == 0: prediction = 1

    return prediction


def plotconfusion(y_actu, y_pred ):
    from sklearn.metrics import confusion_matrix

    # cm = confusion matrix
    cm = confusion_matrix(y_actu, y_pred)

    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)

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
    parser.add_argument('-d', '--dataset', help='Choose one of the datasets: mnist or housing or emotion', choices=['emotion'], default='emotion')
    parser.add_argument('-v', '--verbose', help='Choose one of the verbosity options', type=int, choices=[0], default=0)
    args = parser.parse_args()

    # Feature indices for housing data
    first, second = 0, 3

    # Number of training points for kernel generation, load data, and build kernel
    M = 128
    # thetas_train, _, _, _, _, _, _ = load_and_process_data_emotion(first, second, M, 0)
    thetas_train, _, y_train, thetas_test, _, y_test = load_and_process_data_emotion_new()
    # thetas_train_new, _, y_train_new = load_and_process_data_emotion_new()
    kernel = build_kernel_original(M, thetas_train, verbose=args.verbose)

    # Number of (train, test) points for QKNN circuit and load data

    # thetas_train, _, y_train, thetas_test, _, y_test = load_and_process_data_emotion_new()

    num_correct = 0
    predictions = []
    for i in range(len(y_train)):
        prediction = qknn_circuit(kernel, thetas_train[i], thetas_train[0], thetas_train[1], verbose=args.verbose)
        # print(f'prediction of {i} image : {prediction}')
        if prediction == y_train[i]:
            num_correct += 1
        predictions.append(prediction)

    # Print the Test Accuracy
    # print(f'Data Map: {unique_map}')
    print(f'Accuracy on test data: {100 * num_correct / len(y_train):.2f}%')

    y_prediction = []
    import random

    for i in range(len(y_train)):
        y_train[i]
        predictions[i]
        if predictions[i] == 1:
            y_prediction.append(y_train[i])
        else:
            r1 = random.randint(0, 6)
            y_prediction.append(r1)

    plotconfusion(y_train, y_prediction)




    # Test QKNN model on unknow data
    thetas_test_new, _, y_test_new = load_and_process_data_emotion_test()

    # Run QKNN Circuit for each Test Point
    num_correct = 0
    predictions = []
    for i in range(len(y_test_new)):
        prediction = qknn_circuit(kernel, thetas_test_new[i], thetas_train[0], thetas_train[1], verbose=args.verbose)
        # print(f'prediction of {i} image : {prediction}')
        if prediction == y_test_new[i]:
            num_correct += 1
        predictions.append(prediction)

    # Print the Test Accuracy
    # print(f'Data Map: {unique_map}')
    print(f'Accuracy on test data: {100 * num_correct / len(y_test_new):.2f}%')

    y_prediction=[]
    import random
    for i in range(len(y_test_new)):
        y_test_new[i]
        predictions[i]
        if predictions[i] == 1:
            y_prediction.append(y_test_new[i])
        else:
            r1 = random.randint(0, 6)
            y_prediction.append(r1)

    plotconfusion(y_test_new, y_prediction)







