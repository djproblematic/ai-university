from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier


def evaluate_classifier(classifier, X, y, num_folds=3):
    accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
    precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
    recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
    f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)

    accuracy_mean = round(100 * accuracy_values.mean(), 2)
    precision_mean = round(100 * precision_values.mean(), 2)
    recall_mean = round(100 * recall_values.mean(), 2)
    f1_mean = round(100 * f1_values.mean(), 2)

    return accuracy_mean, precision_mean, recall_mean, f1_mean


input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

classifier_svm = SVC()
classifier_svm.fit(X_train, y_train)

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

num_folds = 3

print('Naive Bayes')
nb_accuracy, nb_precision, nb_recall, nb_f1 = evaluate_classifier(classifier_nb, X, y, num_folds)
print("Accuracy:", nb_accuracy, "%")
print("Precision:", nb_precision, "%")
print("Recall:", nb_recall, "%")
print("F1:", nb_f1, "%")

print('\nSVM')
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_classifier(classifier_svm, X, y, num_folds)
print("Accuracy:", svm_accuracy, "%")
print("Precision:", svm_precision, "%")
print("Recall:", svm_recall, "%")
print("F1:", svm_f1, "%")

visualize_classifier(classifier_nb, X_test, y_test)
visualize_classifier(classifier_svm, X_test, y_test)