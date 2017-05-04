import glob
import os
import random
from os import path

import pandas as pd

import filemanager as fm
import helper

gd_setup = {}
# emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]  # Emotion list
emotions = []
yaml_dict = {}


def get_min_files_in_folder(pv_basepath, check_emotion_list=False):
    ll_folder_names = glob.glob(path.join(pv_basepath, '*'))
    ll_number_of_files = []
    for emotion_folder in ll_folder_names:
        if (check_emotion_list is True and emotion_folder[len(pv_basepath) + 1:] in emotions) \
                or check_emotion_list is False:
            ll_number_of_files.append(len(os.listdir(emotion_folder)))
    return min(ll_number_of_files)


def create_participants_list(pv_basepath, pv_sourcepath):
    ll_participants = []
    emotion_counter = 0
    for emotion in emotions:
        df_filenames = get_filenames(pv_sourcepath, emotion)
        df_filenames["labels"] = emotion_counter
        ll_participants.append(df_filenames)
        emotion_counter += 1
    df_participants = pd.concat(ll_participants)
    df_participants.to_csv(path.join(pv_basepath, "all_participants.csv"), index=False)


def create_evaluation_setup(pv_basepath, pv_sourcepath, pv_fold_number, limit_to_min=False):
    ll_training = []
    ll_testing = []
    emotion_counter = 0
    min_file_number = 0
    if limit_to_min is True:
        min_file_number = get_min_files_in_folder(pv_sourcepath, True)
    for emotion in emotions:
        training, testing = split_training_testing_files(pv_sourcepath, emotion, min_file_number)
        training["labels"] = emotion_counter
        testing["labels"] = emotion_counter
        ll_training.append(training)
        ll_testing.append(testing)
        emotion_counter += 1
    df_training = pd.concat(ll_training)
    df_testing = pd.concat(ll_testing)
    yaml_dict['sampleLimit'] = str(min_file_number)
    df_training.to_csv(path.join(pv_basepath, "fold{0}_train.csv".format(pv_fold_number)), index=False)
    df_testing.to_csv(path.join(pv_basepath, "fold{0}_test.csv".format(pv_fold_number)), index=False)


def split_training_testing_files(pv_sourcepath, emotion,
                                 max_files=0):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob(path.join(pv_sourcepath, "{0}\\*".format(emotion)))
    random.shuffle(files)
    if max_files > 0:
        training = files[:int(max_files * 0.8)]  # get first 80% of file list
        prediction = files[-int(max_files * 0.2):]  # get last 20% of file list
    else:
        training = files[:int(len(files) * 0.8)]  # get first 80% of file list
        prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return pd.DataFrame(training, columns=["filename"]), pd.DataFrame(prediction, columns=["filename"])


def get_filenames(p_path, p_emotion_name):
    types = ('*.jpg', '*.jpeg', '*.png')  # Tuple of file types
    ll_files = []
    for type in types:
        ll_files.extend(glob.glob(path.join("{0}".format(p_path), "{0}".format(p_emotion_name), type)))
    return pd.DataFrame(ll_files, columns=["filename"])


def save_to_yaml(pv_basepath):
    v_yaml_path = path.join(pv_basepath, "config.yml")
    fm.yaml_save_file(v_yaml_path, yaml_dict)


def main():
    global emotions
    gd_setup, __ = helper.load_setup()
    emotions = gd_setup['emotionList']
    v_basepath = gd_setup['evaluationSetupPath']
    v_sourcepath = gd_setup['sourceFilesPath']
    v_folds = gd_setup['evaluationFolds']

    helper.create_directory((v_basepath))
    create_participants_list(pv_basepath=v_basepath, pv_sourcepath=v_sourcepath)
    # create_participants_list(pv_basepath="test_setup2", pv_sourcepath="Data\\sorted_set_testing2")
    for i in xrange(v_folds):
        create_evaluation_setup(pv_basepath=v_basepath, pv_sourcepath=v_sourcepath,
                                pv_fold_number=i, limit_to_min=True)

    # Update yaml dict
    yaml_dict['basepath'] = v_basepath
    yaml_dict['sourcepath'] = v_sourcepath
    yaml_dict['folds'] = v_folds
    yaml_dict['emotions'] = emotions
    save_to_yaml(v_basepath)


if __name__ == "__main__":
    main()
