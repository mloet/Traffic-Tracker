import string
import easyocr
import cv2
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def transform_lbox(c1, c2, lp):
    """
    Predicts the location of license plate based on car location.

    Args:
        c1 (float list): xmin, ymin, xmax, and ymax of the previous bounding box of car.
        c2 (float list): xmin, ymin, xmax, and ymax of the current bounding box of car.
        lp (float list): xmin, ymin, xmax, and ymax of last detected bounding box of license plate.

    Returns:
        float list: xmin, ymin, xmax, and ymax of predicted bounding box for license plate.
    """
    c1center = np.array([(c1[0] + c1[2])/2, (c1[1] + c1[3])/2])
    c2center = np.array([(c2[0] + c2[2])/2, (c2[1] + c2[3])/2])
    translation  = c2center - c1center
    t_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    scalex = (c2[2] - c2[0])/(c1[2] - c1[0])
    scaley = (c2[3] - c2[1])/(c1[3] - c1[1])
    s_matrix = np.array([[scalex, 0, 0], [0, scaley, 0], [0, 0, 1]])

    transform = np.dot(t_matrix, s_matrix)

    normalize = np.array([lp[0], lp[1], 0])
    p1 = np.array([lp[0], lp[1], 1])
    p2 = np.array([lp[2], lp[3], 1])

    t1 = np.dot(transform, p1-normalize)+normalize
    t2 = np.dot(transform, p2-normalize)+normalize

    return [t1[0], t1[1], t2[0], t2[1]]

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Helper function for draw_car.
    """
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

def draw_car(car, frame):
    """
    Draws bounding boxes and visualization for license plates & text. 

    Args:
        car (dict): Tracking information for car and license plate.
        frame (np.array(int)): cv2 framecap

    Returns:
        np.array(int): updated cv2 framecap with information drawn.
    """
    car_x1, car_y1, car_x2, car_y2 = car['car_bbox']
    draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
            line_length_x=200, line_length_y=200)

    # draw license plate
    x1, y1, x2, y2 = car['license_plate_bbox']
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

    # crop license plate
    license_crop = car['license_crop'].reshape(car['license_crop_shape'])

    H, W, _ = license_crop.shape

    try:
        frame[int(car_y1) - H - 100:int(car_y1) - 100,
            int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

        frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
            int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

        (text_width, text_height), _ = cv2.getTextSize(car['license_number'], cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)

        cv2.putText(frame, car['license_number'], (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
    except:
        pass
    return frame


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_plate(license_plates, vehicle):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (list): 2-D list containing the coordinates of detected license plates [x1, y1, x2, y2, score, class_id].
        vehicle (list): List containing coordinates and id of car.

    Returns:
        list: List containing the coordinates of corresponding license plate of input car [x1, y1, x2, y2, score, class_id].
    """
    x1, y1, x2, y2, id = vehicle

    for plate in license_plates:
        xp1, yp1, xp2, yp2, score, lid = plate
        if x1 < xp1 and y1 < yp1 and x2 > xp2 and y2 > yp2:
            return plate
    return None