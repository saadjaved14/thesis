# Script to detect landmarks using Dlib face detecor
import math
from os import path

import cv2
import dlib
import numpy as np

import filemanager as fm

# Setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to detect landmarks on image and return vector points

def get_landmarks(image):
    detections = detector(image, 1)
    landmarks = []
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)  # Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]  # get distance between each point and the central point in both axes
        ycentral = [(y - ymean) for y in ylist]

        if xlist[26] == xlist[
            29]:  # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            anglerelative = (math.atan((z - ymean) / ((w - xmean) + 10e-6)) * 180 / math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

        landmarks.append(landmarks_vectorised)

    if len(detections) < 1:
        landmarks = "error"
    return landmarks


def get_landmarks_dict(pv_path, pl_participants):
    participants_landmarks_filename = pv_path
    if not path.isfile(participants_landmarks_filename):
        d_participants_landmarks = {}
    else:
        d_participants_landmarks = fm.pickle_load_file(participants_landmarks_filename)

    number_of_participants = len(d_participants_landmarks)

    # Compare loaded dictionary with participants list
    # Update missing participants landmarks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for participant in pl_participants:
        # Check if landmarks are already in dictionary
        if participant in d_participants_landmarks:
            print ("Landmarks available: {0}".format(participant))
            pass
        else:
            print participant
            image = cv2.imread(participant)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                print ("Landmarks skipped: {0}".format(participant))
                pass
            else:
                print ("Landmarks updated: {0}".format(participant))
                d_participants_landmarks.update({participant: landmarks_vectorised})

    # Store updated landmarks pickle file
    fm.pickle_save_file(participants_landmarks_filename, d_participants_landmarks)
    print ("New participants added: {0}".format(len(d_participants_landmarks) - number_of_participants))
    return d_participants_landmarks
