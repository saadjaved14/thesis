import cv2
import filemanager as fm
import landmarkDetector as ld
import numpy as np
import pandas as pd
from scipy import stats
import helper


d_setup, __ = helper.load_setup()
#clf_dict = fm.pickle_load_file(d_setup['classifierPath'])
clf_dict = {}
clf_dict_loaded = False
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


def load_clf_dict():
    global clf_dict
    global clf_dict_loaded
    clf_dict = fm.pickle_load_file(d_setup['classifierPath'])
    clf_dict_loaded = True


def preprocessFrame(frame):
    if (len(frame.shape) > 2):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    ret_image = clahe.apply(gray)
    return ret_image


def predict_image(frame):
    if clf_dict_loaded is False:
        load_clf_dict()
    image = preprocessFrame(frame)
    landmark_vector = ld.get_landmarks(image)
    if landmark_vector == "error":
        return False, ["0"]
    prediction_data = np.array(landmark_vector)
    result = []
    for i, clf in clf_dict.iteritems():
        #print "Model ",i, ": ", clf.predict(prediction_data)
        #ll_pred_prob = clf.predict_proba(prediction_data)
        ll_pred = clf.predict(prediction_data)
        result.append(ll_pred)
    mode_result = stats.mode(result)
    return True, mode_result[0][0]


def predict_features(pl_features):
    if clf_dict_loaded is False:
        load_clf_dict()
    prediction_data = np.array(pl_features)
    #ll_pred_prob = []
    d_pred_prob = dict()
    for i, clf in clf_dict.iteritems():
        #ll_pred_prob.append(pd.DataFrame(clf.predict_proba(prediction_data)))
        d_pred_prob.update({i: clf.predict_proba(prediction_data)[0]})
    #df_pred_proba_fold = pd.concat(ll_pred_prob)
    df_pred_proba_fold = pd.DataFrame(d_pred_prob).T
    df_pred_labels_fold = df_pred_proba_fold.apply(lambda x: np.argmax(x), axis=1)
    return df_pred_proba_fold, df_pred_labels_fold