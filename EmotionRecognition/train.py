import copy
import glob
from os import path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

import filemanager as fm
import helper
import landmarkDetector

json_dict = {}
gd_setup = {}


def train(pv_basepath):
    # Read the names of all the _train and _test csv files
    training_filenames = glob.glob(path.join(pv_basepath, "*train.csv"))
    testing_filenames = glob.glob(path.join(pv_basepath, "*test.csv"))

    # Get file names of all the images in data set
    # Find all participants list
    ll_filenames = glob.glob(path.join(pv_basepath, "*all_participants.csv"))
    if len(ll_filenames) == 0:
        print "Participants list not found! Please run evaluation_setup_creator first"
        return "error"

    # Read participants list from CSV
    df_participants_list = pd.read_csv(ll_filenames[0])

    # Get landmarks dictionary
    d_landmarks = landmarkDetector.get_landmarks_dict(gd_setup['landmarksPath'], df_participants_list['filename'])

    # initialize model as dictionary
    model = {}
    model_count = 0
    model_accuray = []

    # update yaml dict
    json_dict['model'] = {
        'name': 'SVM',
        'kernel': 'linear',
        'size': str(len(training_filenames)),
        'accuracy': {}
    }

    # Setup Support Vector Machine
    # Set the classifier as a support vector machines with polynomial kernel
    clf = SVC(kernel='linear', probability=True, tol=1e-3)  # ,verbose = True)

    # Iterate through each _train file and create a model
    for filename in training_filenames:
        print "Filename: {0}".format(filename)
        df_training = pd.read_csv(filename)
        training_data = []
        training_labels = []
        for index, item in df_training.iterrows():
            if str(item['filename']) in d_landmarks:
                landmarks_vectorised = d_landmarks[str(item['filename'])]
                training_data.append(landmarks_vectorised[0])  # append image array to training data list
                training_labels.append(item["labels"])

        print "Training Model: {0}".format(model_count)
        npar_train = np.array(training_data)
        clf.fit(npar_train, training_labels)

        # Test model for accuracy
        print "Testing Model: {0}".format(testing_filenames[model_count])
        df_testing = pd.read_csv(testing_filenames[model_count])
        testing_data = []
        testing_labels = []
        for index, item in df_testing.iterrows():
            # if d_landmarks.has_key(str(item['filename'])):
            if str(item['filename']) in d_landmarks:
                landmarks_vectorised = d_landmarks[str(item['filename'])]
                testing_data.append(landmarks_vectorised[0])  # append image array to training data list
                testing_labels.append(item["labels"])
        npar_test = np.array(testing_data)
        test_score = clf.score(npar_test, testing_labels)
        print "Accuracy: {0}".format(test_score)
        model_accuray.append(test_score)

        # update yaml dict for accuracies
        json_dict['model']['accuracy'].update({str(model_count): str(test_score)})

        # Update trained model
        model.update({str(model_count): copy.deepcopy(clf)})
        model_count += 1

    # Find mean accuracy of the model
    print "Mean accuracy: {0}".format(np.mean(model_accuray))
    json_dict['model']['meanAccuracy'] = str(np.mean(model_accuray))

    # Save model to file
    fm.pickle_save_file(gd_setup['classifierPath'], model)
    print "Model Saved."


def main():
    global gd_setup
    gd_setup, __ = helper.load_setup()
    train(gd_setup['evaluationSetupPath'])
    fm.json_save_file(path.join(gd_setup['evaluationSetupPath'], 'results.json'), json_dict)


if __name__ == "__main__":
    main()
