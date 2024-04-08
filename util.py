import string
import easyocr
import cv2
from paddleocr import PaddleOCR

# from matplotlib import pyplot as plt


# Initialize the OCR reader
reader = PaddleOCR(lang='en', use_gpu=True)

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


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write(
            '{},{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'car_score', 'car_name',
                                                  'license_plate_bbox',
                                                  'license_plate_bbox_score',
                                                  'license_number',
                                                  'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    # Get other information
                    car_bbox = results[frame_nmr][car_id]['car']['bbox']
                    car_score = results[frame_nmr][car_id]['car']['score']
                    car_name = results[frame_nmr][car_id]['car']['name']
                    license_plate_bbox = results[frame_nmr][car_id]['license_plate']['bbox']
                    license_plate_bbox_score = results[frame_nmr][car_id]['license_plate']['bbox_score']
                    license_number = results[frame_nmr][car_id]['license_plate']['text']
                    license_number_score = results[frame_nmr][car_id]['license_plate']['text_score']

                    # Write to CSV
                    f.write('{},{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                                  car_id,
                                                                  '[{} {} {} {}]'.format(
                                                                      car_bbox[0], car_bbox[1], car_bbox[2],
                                                                      car_bbox[3]),
                                                                  car_score,
                                                                  car_name,
                                                                  '[{} {} {} {}]'.format(
                                                                      license_plate_bbox[0], license_plate_bbox[1],
                                                                      license_plate_bbox[2], license_plate_bbox[3]),
                                                                  license_plate_bbox_score,
                                                                  license_number,
                                                                  license_number_score))
    f.close()


# def license_complies_format(text):
#     if len(text) != 10:
#         return False
#
#     if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
#             (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
#             (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
#             (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
#             (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
#             (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
#             (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
#             (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()) and \
#             (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in dict_char_to_int.keys()) and \
#             (text[9] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[9] in dict_char_to_int.keys()):
#         return True
#     else:
#         return False
#
#
# def format_license(text):
#     license_plate_ = ''
#     mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_char_to_int,
#                2: dict_char_to_int, 3: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int, 9: dict_char_to_int}
#     for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         if text[j] in mapping[j].keys():
#             license_plate_ += mapping[j][text[j]]
#         else:
#             license_plate_ += text[j]
#
#     return license_plate_

def license_complies_format(text):
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
    detections = reader.ocr(license_plate_crop)
    if detections[0]:
        detections = detections[0]
        text = [line[1][0] for line in detections]
        score = [line[1][1] for line in detections]

        if text:
            text = text[0].upper().replace(' ', '')
            text = text.replace('.', '')

        if license_complies_format(text):
            return format_license(text), score[0]

        return None, None
    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, score, car_id, car_class_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1, -1, -1


def gray_image(image):
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return processed_image


def blurred_image(image):
    processed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return processed_image


def filtered_image(image):
    processed_image = cv2.medianBlur(image, 3)
    return processed_image


def threshold_image_1(image):
    _, threshold_image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY_INV)
    return threshold_image


def threshold_image_2(image):
    _, threshold_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                               11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    return processed_image


def denoised_image(image):
    processed_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7,
                                                      21)
    return processed_image


def forallisone(license_plate_crop):
    license_plate_crop_gray = gray_image(license_plate_crop)
    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
    if license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
    if license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(blurred_image(license_plate_crop_gray))
    if license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(filtered_image(license_plate_crop_gray))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(threshold_image_1(license_plate_crop_gray))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(threshold_image_2(license_plate_crop_gray))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(denoised_image(license_plate_crop_gray))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(
            threshold_image_1(filtered_image(blurred_image(license_plate_crop_gray))))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(
            threshold_image_1(blurred_image(license_plate_crop_gray)))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(
            threshold_image_2(blurred_image(license_plate_crop_gray)))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(
            threshold_image_1(filtered_image(license_plate_crop_gray)))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(
            threshold_image_2(filtered_image(license_plate_crop_gray)))
    elif license_plate_text is None:
        license_plate_text, license_plate_text_score = read_license_plate(
            threshold_image_2(filtered_image(blurred_image(license_plate_crop_gray))))
    else:
        license_plate_text, license_plate_text_score = read_license_plate(
            filtered_image(blurred_image(license_plate_crop_gray)))

    return license_plate_text, license_plate_text_score
