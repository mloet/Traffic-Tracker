from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate
import numpy as np


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

mot_tracker = Sort()

# load models
coco_model = YOLO('./yolov8n.pt')
license_plate_detector = YOLO('./license_plate.pt')

# load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_nmr = 0
# read frames
ret = True

car_dict = {}
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        detected = {}
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
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

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

                    detected[car_id] = {'car_id': car_id, 
                          'car_bbox': [xcar1, ycar1, xcar2, ycar2], 
                          'license_plate_bbox': [x1, y1, x2, y2],
                          'license_number': license_plate_text,
                          'license_plate_bbox_score': score,
                          'license_number_score': license_plate_text_score,
                          'license_crop': license_crop,
                          'license_crop_shape': (400, license_crop_shape, 3)
                          }
            

        
        if detected:
          for car in detected:
              # draw car
              car_x1, car_y1, car_x2, car_y2 = detected[car]['car_bbox']
              draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                          line_length_x=200, line_length_y=200)

              # draw license plate
              x1, y1, x2, y2 = detected[car]['license_plate_bbox']
              cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

              # crop license plate
              license_crop = detected[car]['license_crop'].reshape(detected[car]['license_crop_shape'])

              H, W, _ = license_crop.shape

              try:
                  frame[int(car_y1) - H - 100:int(car_y1) - 100,
                        int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                  frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                        int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                  (text_width, text_height), _ = cv2.getTextSize(
                      detected[car]['license_number'],
                      cv2.FONT_HERSHEY_SIMPLEX,
                      4.3,
                      17)

                  cv2.putText(frame,
                              detected[car]['license_number'],
                              (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              4.3,
                              (0, 0, 0),
                              17)

              except:
                  pass

        frame = cv2.resize(frame, (1280, 720))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
          break

cap.release()

