from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix


def linear_classifier(
    features_train,
    y_train,
    features_test,
    y_test,
    class_labels,
    fraction, #fraction=1.0,
    test_size=0.2,
):
    """Evaluates the feature quality by the means of a logistic regression classifier
    Args:
        features: Training instances
        y: labels (ints)
        class_labels: labels (strings)
        fraction: fraction of features used for training clf
        test_size: fraction of features used for evaluation clf
    Prints:
        Top-1 accuracy and classification reports
    Notes:
        10-fold cross-validation is used to tune regularization hyperparameters of clf
    """
    if fraction != 1.0:
        features_train, feat_not_used, y_train, y_not_used = train_test_split(
            features_train,
            y_train,
            test_size=1 - fraction / (1 - test_size),
            random_state=42,
            shuffle=True,
        )

    clf = LogisticRegressionCV(cv=5, max_iter=1000, verbose=0, n_jobs=8).fit(
        features_train, y_train
    )

    # Evaluate on test
    print(f"Accuracy on test: {round(clf.score(features_test, y_test),2)} \n")
    y_pred_test = clf.predict(features_test)
    classification_report_test = classification_report(
        y_test,
        y_pred_test,
        labels=list(range(0, len(class_labels))),
        target_names=class_labels,
    )
    print(classification_report_test)

    return y_pred_test


if __name__ == "__main__":

    #导入数据集
    X = sio.loadmat('data/test/X_tf_train_416.mat')
    features_train = X['X_tf_train_416']
    del X
    X = sio.loadmat('data/test/Y_tf_train_416.mat')
    y = X['Y_tf_train_416']
    y_train = np.ones(y.shape[0])
    y_train = y[0, :]
    del X, y
    X = sio.loadmat('data/test/X_tf_test_4780.mat')
    features_test = X['X_tf_test_4780']
    del X
    X = sio.loadmat('data/test/Y_tf_test_4780.mat')
    y = X['Y_tf_test_4780']
    y_test = np.ones(y.shape[0])
    y_test = y[0, :]

    class_labels = ['label_' + str(i) for i in range(2)]

    fractions = [1.0]
    for fraction in fractions:
        print(f"    ==== Linear {fraction * 100}% of the training data used ==== \n")
        y_pre = linear_classifier(features_train, y_train, features_test, y_test, class_labels, fraction = fraction)


    CM = confusion_matrix(y_test, y_pre)

