import predictor
import pandas as pd
from os import path
import glob
import cv2
import metricscalc as mc


def test(pv_path):
    true_labels = []
    predicted_labels = []

    ll_participants_filename = glob.glob(path.join(pv_path, "*.csv"))

    # Load participant names and labels from csv
    df_participants = pd.read_csv(ll_participants_filename[0])

    # Predict labels for all participants
    for index, item in df_participants.iterrows():
            print "File name: {0}".format(item['filename'])
            print "True label: {0}".format(item['labels'])
            img = cv2.imread(item['filename'])
            result_status, pred_result = predictor.predict_image(img)
            if result_status is False:
                print "No face detected"
                pass
            else:
                print "Predicted label: {0}".format(pred_result[0])
                true_labels.append(item['labels'])
                predicted_labels.append(pred_result[0])
    df_results = pd.DataFrame(true_labels, columns=['true'])
    df_results['predicted'] = predicted_labels
    df_results.to_csv(path.join(pv_path, "label_results.csv"), index=False)


def confusion_matrix(pv_path):
    df_labels = pd.read_csv(path.join(pv_path, "label_results.csv"))
    ll_emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    mc.create_confusion_matrix(pv_path, df_labels['true'], df_labels['predicted'], ll_emotions)


if __name__ == "__main__":
    test(pv_path="test_setup1")
    confusion_matrix("test_setup1")
