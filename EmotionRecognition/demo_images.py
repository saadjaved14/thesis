import cv2
import predictor
import overlay_helper
import helper
import glob
from os import path

d_setup, d_config = helper.load_setup()

def get_image_list(pv_path):
    ll_files = []
    for img_format in ('*.png', '*.jpg', '*.jpeg'):
        ll_files.extend(glob.glob(path.join(pv_path, img_format)))
    return ll_files


def process(pv_path):
    ll_image_files = get_image_list(pv_path)
    for img_path in ll_image_files:
        if '_result' in img_path:
            continue
        print img_path

        # Read image
        img = cv2.imread(img_path)

        # Predict face emotions on frame
        b_found, ll_pred, ll_pred_label, llol_pred_prob, landmarks = predictor.predict_image(img)

        # Display results
        if b_found is True:
            # print ll_pred_label
            # print llol_pred_prob
            face_count = 0
            for label in ll_pred_label:
                # cv2.putText(img, label, landmarks[face_count][0], cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                face_count += 1

            # Display landmarks
            landmark_count = 1
            for points in landmarks[0]:
                if landmark_count in (36, 45):
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.circle(img, points, 5, color, -1)
                landmark_count += 1

            # Draw bars
            overlay_helper.draw_bars(img, llol_pred_prob, landmarks[0][0], d_config['emotions'])

        # Write output
        cv2.imwrite(img_path[:-4] + '_result.png', img)

        # Hold results for diplay
        #cv2.imshow('Results', img)
        #cv2.waitKey(0)



if __name__ == '__main__':
    process('E:\\00_TH\\DummyData\\images')