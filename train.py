import cv2
import glob
import numpy as np
import landmarkDetector
import copy
import filemanager as fm
from sklearn.svm import SVC
from os import path
import pandas as pd
import helper


json_dict = {}
gd_setup = {}


def get_landmarks_dict(pv_basepath):
    # Find all participants list
    ll_filenames = glob.glob(path.join(pv_basepath, "*all_participants.csv"))
    if len(ll_filenames) == 0:
        print "Participants list not found! Please run evaluation_setup_creator first"
        return "error"

    # Read participants list from CSV
    df_participants_list = pd.read_csv(ll_filenames[0])

    # Find dictionary pickle file and load it
    ll_landmarks_file = glob.glob(path.join(gd_setup['landmarksPath'], "*landmark.pickle"))
    if len(ll_landmarks_file) == 0:
        d_participants_landmarks = {}
        participants_landmarks_filename = path.join(pv_basepath, "participants_landmark.pickle")
    else:
        d_participants_landmarks = fm.pickle_load_file(ll_landmarks_file[0])
        participants_landmarks_filename = ll_landmarks_file[0]

    number_of_participants = len(d_participants_landmarks)

    # Compare loaded dictionary with participants list
    # Update missing participants landmarks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for index, item in df_participants_list.iterrows():
        # Check if landmarks are already in dictionary
        if str(item['filename']) in d_participants_landmarks:
            print ("Landmarks available: {0}".format(str(item['filename'])))
            pass
        else:
            image = cv2.imread(item["filename"])            # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = landmarkDetector.get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                print ("Landmarks skipped: {0}".format(str(item['filename'])))
                pass
            else:
                print ("Landmarks updated: {0}".format(str(item['filename'])))
                d_participants_landmarks.update({str(item["filename"]): landmarks_vectorised})

    # Store updated landmarks pickle file
    fm.pickle_save_file(participants_landmarks_filename, d_participants_landmarks)
    print ("New participants added: {0}".format(len(d_participants_landmarks) - number_of_participants))
    return d_participants_landmarks


def train(pv_basepath):
    # Read the names of all the _train and _test csv files
    training_filenames = glob.glob(path.join(pv_basepath, "*train.csv"))
    testing_filenames = glob.glob(path.join(pv_basepath, "*test.csv"))

    # Get landmarks dictionary
    d_landmarks = get_landmarks_dict(pv_basepath)

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
