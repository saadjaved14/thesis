import cv2
import predictor
import pandas as pd
import helper

d_setup, d_config = helper.load_setup()

def start_capture():
    # Create video capture
    #cv_capture = cv2.VideoCapture('D:\\Documents\\Thesis\\Raw Data\\Test_Video\\2.avi')
    cv_capture = cv2.VideoCapture(0)
    ret, frame = cv_capture.read()
    height, width, layers = frame.shape
    fourcc = -1 #cv2.VideoWriter_fourcc(*'XVID')
    # cv_writer = cv2.VideoWriter('D:\\Documents\\Thesis\\Raw Data\\Test_Video\\live_result.avi', fourcc, 20, (width, height))
    df_history = pd.DataFrame()
    v_frame_number = 0
    while True:
        # Capture frame by frame
        ret, frame = cv_capture.read()

        # Predict face emotions on frame
        b_found, ll_pred, ll_pred_label, llol_pred_prob, landmarks = predictor.predict_image(frame)

        # Display results
        if b_found is True:
            print ll_pred_label
            print llol_pred_prob
            face_count = 0
            for label in ll_pred_label:
                cv2.putText(frame, label, landmarks[face_count][0], cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                face_count += 1

        # Save history
        df_history.set_value(int(v_frame_number), 'frame', int(v_frame_number))
        df_history.set_value(int(v_frame_number), 'label', ll_pred_label[0])
        emotion_count = 0
        for emotion in d_config['emotions']:
            df_history.set_value(int(v_frame_number), emotion, llol_pred_prob[0][emotion_count])
            emotion_count += 1

        cv2.imshow('Live', frame)
        #cv_writer.write(frame)
        v_frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv_capture.release()
    cv2.destroyWindow('Live')
    #cv_writer.release()
    df_history.to_csv('D:\\Documents\\Thesis\\Raw Data\\Test_Video\\results_nodisgust.csv')


if __name__ == '__main__':
    start_capture()