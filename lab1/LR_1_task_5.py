import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_curve, \
    roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
df.head()

df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

confusion_matrix(df.actual_label.values, df.predicted_RF.values)


def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def davydov_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


davydov_confusion_matrix(df.actual_label.values, df.predicted_RF.values)

assert np.array_equal(davydov_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values))


def davydov_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FP + TN + FN)


assert davydov_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values,
                                                                                                  df.predicted_RF.values), 'davydov_accuracy_score failed on RF'

assert davydov_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values,
                                                                                                  df.predicted_LR.values), 'davydov_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (davydov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (davydov_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

accuracy_score(df.actual_label.values, df.predicted_RF.values)
recall_score(df.actual_label.values, df.predicted_RF.values)


def davydov_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert davydov_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values,
                                                                                              df.predicted_RF.values), 'davydov_recall_score failed on RF'

assert davydov_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values,
                                                                                              df.predicted_LR.values), 'davydov_recall_score failed on LR'

print('Recall RF: %.3f' % (davydov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (davydov_recall_score(df.actual_label.values, df.predicted_LR.values)))

precision_score(df.actual_label.values, df.predicted_RF.values)


def davydov_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert davydov_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values), 'davydov_precision_score failed on RF'

assert davydov_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values), 'davydov_precision_score failed on LR'

print('Precision RF: %.3f' % (davydov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (davydov_precision_score(df.actual_label.values, df.predicted_LR.values)))

f1_score(df.actual_label.values, df.predicted_RF.values)


def davydov_f1_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    recall = davydov_recall_score(y_true, y_pred)
    precision = davydov_precision_score(y_true, y_pred)
    return 2 * (recall * precision) / (recall + precision)


assert davydov_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values,
                                                                                      df.predicted_RF.values), 'davydov_f1_score failed on RF'
assert davydov_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values,
                                                                                      df.predicted_LR.values), 'davydov_f1_score failed on LR'

print('F1 RF: %.3f' % (davydov_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (davydov_f1_score(df.actual_label.values, df.predicted_LR.values)))

print('\nscores with threshold = 0.5')
print('Accuracy RF: %.3f' % (davydov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (davydov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (davydov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (davydov_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (davydov_accuracy_score(df.actual_label.values, df.predicted_LR.values)))
print('Recall LR: %.3f' % (davydov_recall_score(df.actual_label.values, df.predicted_LR.values)))
print('Precision LR: %.3f' % (davydov_precision_score(df.actual_label.values, df.predicted_LR.values)))
print('F1 LR: %.3f' % (davydov_f1_score(df.actual_label.values, df.predicted_LR.values)))
print('')

print('scores with threshold = 0.25')
print(
    'Accuracy RF: %.3f' % (davydov_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (davydov_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (
    davydov_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (davydov_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print(
    'Accuracy LR: %.3f' % (davydov_accuracy_score(df.actual_label.values, (df.model_LR >= 0.25).astype('int').values)))
print('Recall LR: %.3f' % (davydov_recall_score(df.actual_label.values, (df.model_LR >= 0.25).astype('int').values)))
print('Precision LR: %.3f' % (
    davydov_precision_score(df.actual_label.values, (df.model_LR >= 0.25).astype('int').values)))
print('F1 LR: %.3f' % (davydov_f1_score(df.actual_label.values, (df.model_LR >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
