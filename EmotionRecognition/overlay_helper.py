import cv2


def draw_bars(p_frame, pl_values, pl_point, pl_labels):
    if p_frame.size == 0:
        return p_frame

    v_offset = 10
    v_text_offset = 50
    v_rect_height = 5
    index_lvalues = 0
    for ll_values in pl_values:
        index_values = 0
        for values in ll_values:
            y_start = pl_point[index_lvalues][1] + v_offset * index_values
            x_bar_start = pl_point[index_lvalues][0] + v_text_offset

            # Check for bounds
            frame_height = p_frame.shape[0]
            frame_width = p_frame.shape[1]

            if (pl_point[index_lvalues][0] > frame_width
                or  y_start + v_rect_height > frame_height
                or x_bar_start + int(values * 100) > frame_width):
                continue

            cv2.putText(p_frame, pl_labels[index_values], (pl_point[index_lvalues][0], y_start + v_rect_height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.rectangle(p_frame, (x_bar_start, y_start),
                          (x_bar_start + int(values * 100), y_start + v_rect_height), 255, -1)
            index_values += 1
        index_lvalues += 1

    return p_frame


def get_draw_points_from_landmarks(landmarks, v_landmark_index):
    ll_ret_points = []
    for face_points in landmarks:
        ll_ret_points.append(face_points[v_landmark_index])
    return ll_ret_points