import itertools
from os import path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

'''def compute_metrics_cm(self, pma_true, pf_result, pv_label, pv_multilabel):
    """
    Compute confusion matrix based metrics
    :param pma_true: np.ndarray
        ground truth labels
    :param pma_predicted: np.ndarray
        predicted labels
    :return:
    """
    lma_predicted = pf_result[pv_label]

    # threshold to get the labels
    if pv_multilabel:
        lma_predicted[lma_predicted >= self.cd_evaluation['threshold'][self.cd_classifier['method']]] = 1
        lma_predicted[lma_predicted < self.cd_evaluation['threshold'][self.cd_classifier['method']]] = 0
    else:
        lma_predicted = pf_result[self.cd_classifier_parameters['labels'].keys()].apply(lambda x: argmax(x), axis=1)
        lma_predicted = lma_predicted.apply(lambda x: 1 if x == pv_label else 0)
    lma_cm = ConfusionMatrix(pma_true, lma_predicted)._df_confusion
    # ld_results = lma_cm.stats()
    ld_metrics = {}

    # False rejection rate
    lv_frr = lma_cm.ix[1, 0] / float(lma_cm.ix[1, :].sum())
    ld_metrics.update({'false rejection rate': lv_frr})

    # False acceptance rate
    lv_far = lma_cm.ix[0, 1] / float(lma_cm.ix[0, :].sum())
    ld_metrics.update({'false acceptance rate': lv_far})

    # Specificity
    lv_specificity = lma_cm.ix[0, 0] / float(lma_cm.ix[:, 0].sum())
    ld_metrics.update({'Specificity': lv_specificity})

    # Sensitivity
    lv_sensitivity = lma_cm.ix[1, 1] / float(lma_cm.ix[:, 1].sum())
    ld_metrics.update({'Precision': lv_sensitivity})

    # Recall
    lv_recall = lma_cm.ix[1, 1] / float(lma_cm.ix[1, :].sum())
    ld_metrics.update({'recall': lv_recall})

    # Accuracy
    lv_accuracy = (lma_cm.ix[1, 1] + lma_cm.ix[0, 0]) / float(lma_cm.sum().sum())
    ld_metrics.update({'Accuracy': lv_accuracy})

    # F1 score
    lv_f1 = 2 * lv_sensitivity * lv_recall / float(lv_sensitivity + lv_recall)
    ld_metrics.update({'f1': lv_f1})

    ''''''ld_metrics = {'precision': ld_results['class'].ix['PPV: Pos Pred Value (Precision)', 1.0],
                  'sensitivity': ld_results['class'].ix['TPR: (Sensitivity, hit rate, recall)', 1.0],
                    'f1_score': ld_results['class'].ix['F1 score', 1.0]}''''''
    return ld_metrics
    '''


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            value = '{:.2f}'.format(cm[i, j])
        else:
            value = '{0}'.format(cm[i, j])
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_confusion_matrix(pv_path, pa_y_true, pa_y_pred, pa_labels, fold=''):
    nparr_true = np.array(pa_y_true)
    nparr_pred = np.array(pa_y_pred)
    nparr_labels = np.array(pa_labels)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(nparr_true, nparr_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=nparr_labels,
                          title='Confusion matrix, without normalization')
    current_path = path.join(pv_path, "cnf_mtx_fold_{0}.png".format(fold))
    plt.savefig(current_path)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=nparr_labels, normalize=True,
                          title='Normalized confusion matrix')
    current_path = path.join(pv_path, "cnf_mtx_nml_fold_{0}.png".format(fold))
    plt.savefig(current_path)
    plt.clf()
    # plt.show()
