import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'

data = np.loadtxt(input_file, delimiter = ',')

X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]

print(accuracy)
print("Accuracy of the new Naive Bayes classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier, X_test, y_test)

num_folds = 3
accuracy_values = cross_val_score(classifier, X, y, scoring = 'accuracy', cv = num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")


# import numpy as np
# from sklearn.naive_bayes import GaussianNB
# from utilities import visualize_classifier

# input_file = 'data_multivar_nb.txt'

# data = np.loadtxt(input_file, delimiter=',')
# X, y = data[:, :-1], data[:, -1]

# classifier = GaussianNB()
# classifier.fit(X, y)

# y_pred = classifier.predict(X)
# accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]

# print(accuracy)
# print("Accuracy of the Naive Bayes classifier =", round(accuracy, 2), "%")

# visualize_classifier(classifier, X, y)