import ast

import cv2
import numpy as np
import pandas as pd
import random

def draw_border(car_name, img, top_left, bottom_right, color=(0, 255, 0), thickness=12, line_length_x=150,
                line_length_y=150,):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    # cv2.putText(img, car_name, (x1+200, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), thickness-3)
    cv2.putText(img, car_name, (x1+190, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
    return img


results = pd.read_csv('./test_interpolated.csv')

# load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
car = {}
car_colors = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    max_2 = np.amax(results[results['car_id'] == car_id]['car_score'])
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    car_colors[car_id] = color


    car[car_id] = {
        'name': results[(results['car_id'] == car_id) & (results['car_score'] == max_2)]['car_name'].iloc[0]}

    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)][
                                 'license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[
                                          0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                               ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 100 / (y2 - y1)), 100))

    license_plate[car_id]['license_crop'] = license_crop

frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]

        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                      ','))
            color = car_colors[df_.iloc[row_indx]['car_id']]
            draw_border(car[df_.iloc[row_indx]['car_id']]['name'], frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)),
                        color=color,
                        line_length_x=100, line_length_y=100)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ',
                                                                                                        ' ').replace(
                    ' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255),4)

            # crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                frame[int(car_y1) - H:int(car_y1),
                int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H-H:int(car_y1) - H,
                int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    6)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width + 160) / 2), int(car_y1 - 1.5 * H + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.25,
                            (0, 0, 0),
                            6)

            except:
                pass

        out.write(frame)
        frame = cv2.resize(frame, (1920, 1080))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

out.release()
cap.release()