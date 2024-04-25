from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_plate, read_license_plate, draw_car, transform_lbox
import numpy as np
import pandas as pd

mot_tracker = Sort()

# load models
coco_model = YOLO('./yolov8n.pt')
license_plate_detector = YOLO('./license_plate.pt')

# load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
car_dict = {}
while ret:
    ret, frame = cap.read()
    display = frame
    if ret:
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for i, car in enumerate(track_ids):
            car_bbox = car[:4]
            car_id = car[4]

            plate = get_plate(license_plates.boxes.data.tolist(), car)

            # match detected plates with cars
            if plate and car_id != -1:
                x1, y1, x2, y2, score, class_id = plate
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    license_crop_shape = int((x2 - x1) * 400 / (y2 - y1))
                    license_crop = cv2.resize(license_plate_crop, (license_crop_shape, 400)).flatten()

                    # check plate number with highest confidence score
                    if car_id in car_dict:
                        if license_plate_text_score < car_dict[car_id]['license_number_score']:
                            license_plate_text, license_plate_text_score = car_dict[car_id]['license_number'], car_dict[car_id]['license_number_score']

                    
                    info = {'car_bbox': car_bbox, 
                            'license_plate_bbox': [x1, y1, x2, y2],
                            'license_number': license_plate_text,
                            'license_plate_bbox_score': score,
                            'license_number_score': license_plate_text_score,
                            'license_crop': license_crop,
                            'license_crop_shape': (400, license_crop_shape, 3)
                            }

                    display = draw_car(info, display)
                    car_dict[car_id] = info

            # interpolate location of previously recognized car & plate pairing
            elif car_id in car_dict and car_id != -1:
                info = car_dict[car_id] 
                info['license_plate_bbox'] = transform_lbox(info['car_bbox'], car_bbox, info['license_plate_bbox'])
                info['car_bbox'] = car_bbox

                display = draw_car(info, display)
                car_dict[car_id] = info
        
        
        display = cv2.resize(display, (1280, 720))

        cv2.imshow('display', display)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()
