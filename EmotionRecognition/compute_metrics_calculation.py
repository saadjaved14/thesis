import collections
import glob
from os import path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import *

import helper
import landmarkDetector
import metricscalc as mc
import predictor

gl_emotions = []
gd_setup = {}
gd_config = {}


def plot_curve(pv_path, pd_result, pl_labels, pv_title, pv_fold=-1):
    for fold in xrange(len(pd_result)):
        if pv_fold >= 0 and fold == pv_fold:
            for emotion in gl_emotions:
                plt.plot(pd_result[fold][emotion][pl_labels[0]], pd_result[fold][emotion][pl_labels[1]], label=emotion)
                plt.title('{0} Fold {1}'.format(pv_title, fold))
                plt.xlabel(pl_labels[0])
                plt.ylabel(pl_labels[1])
                plt.legend()
            plt.savefig(path.join(pv_path, '{0}_fold_{1}.png'.format(pv_title, fold)))
            plt.clf()


def compute_confusion_matrix(pdf_groundtruth, pd_predicted, pv_path, pv_fold=-1):
    for fold in xrange(len(pd_predicted)):
        if pv_fold >= 0 and fold == pv_fold:
            y_pred = []
            y_gt = []
            for participant_idx, participant in pdf_groundtruth.iterrows():
                if participant['filename'] in pd_predicted[str(fold)].keys():
                    y_pred.append(pd_predicted[str(fold)][participant['filename']])
                    y_gt.append(participant['labels'])
            mc.create_confusion_matrix(pv_path, y_gt, y_pred, gl_emotions, str(fold))


def compute_precision_recall_curve(pl_participant, pd_predicted, pd_groundtruth):
    d_avg_precision = dict()
    d_prc = helper.nested_dict()
    for fold in xrange(len(pd_predicted)):
        for emotion in xrange(len(gl_emotions)):
            y_pred = []
            y_gt = []
            for participant in pl_participant:
                if participant in pd_predicted[str(fold)].keys():
                    y_gt.append(pd_groundtruth[participant][emotion])
                    y_pred.append(pd_predicted[str(fold)][participant].ix[emotion])

            precision, recall, __ = precision_recall_curve(y_gt, y_pred)
            d_prc[fold][gl_emotions[emotion]]['precision'] = precision
            d_prc[fold][gl_emotions[emotion]]['recall'] = recall
    return d_prc


def compute_roc_curve(pl_participant, pd_predicted, pd_groundtruth):
    d_roc = helper.nested_dict()
    for fold in xrange(len(pd_predicted)):
        for emotion in xrange(len(gl_emotions)):
            y_pred = []
            y_gt = []
            for participant in pl_participant:
                if participant in pd_predicted[str(fold)].keys():
                    y_gt.append(pd_groundtruth[participant][emotion])
                    y_pred.append(pd_predicted[str(fold)][participant].ix[emotion])

            fpr, tpr, __ = roc_curve(y_gt, y_pred)
            d_roc[fold][gl_emotions[emotion]]['fpr'] = fpr
            d_roc[fold][gl_emotions[emotion]]['tpr'] = tpr
    return d_roc


def get_participants(pv_path):
    participants_list_path = glob.glob(path.join(pv_path, '*participants.csv'))
    df_participants = pd.read_csv(participants_list_path[0])
    return df_participants


def get_prediction_probabilities(pl_participants, pd_landmarks):
    d_probabilities = collections.defaultdict(dict)
    d_labels = collections.defaultdict(dict)
    for participant in pl_participants:
        if participant in pd_landmarks:
            df_pred_result, df_pred_labels = predictor.predict_features(pd_landmarks[participant])
            for rowidx, row in df_pred_result.iterrows():
                d_probabilities[rowidx][participant] = row
                d_labels[rowidx][participant] = df_pred_labels[rowidx]
                # d_probabilities.update({participant: predictor.predict_features(pd_landmarks[participant])})
    return dict(d_probabilities), dict(d_labels)


def create_ground_truth_dict(pdf_participants):
    d_groundtruth = dict()
    for index, participant in pdf_participants.iterrows():
        ll_binary_emotion_label = [0] * len(gl_emotions)
        ll_binary_emotion_label[int(participant['labels'])] = 1
        d_groundtruth.update({participant['filename']: ll_binary_emotion_label})
    return d_groundtruth


def run_main_loop(pv_testpath):
    ll_fold_names = glob.glob(path.join(pv_testpath, '*'))
    # ll_fold_names = ['test_setup2']
    lv_fold = 0
    for folder_name in ll_fold_names:
        df_participants = get_participants(folder_name)
        d_participants_landmarks = landmarkDetector.get_landmarks_dict(gd_setup['landmarksPath'],
                                                                       df_participants['filename'])
        d_pred_prob, d_pred_labels = get_prediction_probabilities(df_participants['filename'], d_participants_landmarks)
        d_ground_truth = create_ground_truth_dict(df_participants)
        compute_confusion_matrix(df_participants, d_pred_labels, folder_name, lv_fold)
        d_roc = compute_roc_curve(df_participants['filename'], d_pred_prob, d_ground_truth)
        d_prc = compute_precision_recall_curve(df_participants['filename'], d_pred_prob, d_ground_truth)
        plot_curve(folder_name, d_roc, ['fpr', 'tpr'], 'roc', lv_fold)
        plot_curve(folder_name, d_prc, ['recall', 'precision'], 'prc', lv_fold)
        lv_fold += 1


def main():
    global gl_emotions
    global gd_setup
    global gd_config
    gd_setup, gd_config = helper.load_setup()
    gl_emotions = gd_config['emotions']
    run_main_loop(gd_setup['testSetupPath'])
    print "Task completed"


if __name__ == '__main__':
    main()
