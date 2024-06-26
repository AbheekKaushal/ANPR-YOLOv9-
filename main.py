import numpy as np
from ultralytics import YOLO
import cv2
from tracker import Tracker
from util import get_car, read_license_plate, write_csv,forallisone

results = {}

tracker = Tracker()

# load models
coco_model = YOLO('yolov9e.pt')
license_plate_detector = YOLO('models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

coco_names = {
    2: "CAR",
    3: "BIKE",
    5: "BUS",
    7: "TRUCK"
}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        track_ids = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score, class_id])

        tracker.update(frame, np.asarray(detections_))
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            track_ids.append([x1, y1, x2, y2, track.score, track.track_id, track.class_id])

        # detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_score, car_id, car_class_id = get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                license_plate_text, license_plate_text_score = forallisone(license_plate_crop)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2],
                                                          'score': car_score,
                                                          'name': coco_names[int(car_class_id)]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test.csv')

