import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

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
        f.write('{},{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'class_name', 'car_bbox', 'car_score',
                                                      'license_plate_bbox', 'license_plate_bbox_score',
                                                      'license_number',
                                                      'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    # Get class name
                    class_name = results[frame_nmr][car_id]['class_name']

                    # Get other information
                    car_bbox = results[frame_nmr][car_id]['car']['bbox']
                    car_score = results[frame_nmr][car_id]['car']['score']
                    license_plate_bbox = results[frame_nmr][car_id]['license_plate']['bbox']
                    license_plate_bbox_score = results[frame_nmr][car_id]['license_plate']['bbox_score']
                    license_number = results[frame_nmr][car_id]['license_plate']['text']
                    license_number_score = results[frame_nmr][car_id]['license_plate']['text_score']

                    # Write to CSV
                    f.write('{},{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                               car_id,
                                                               class_name,
                                                               '[{} {} {} {}]'.format(
                                                                   car_bbox[0], car_bbox[1], car_bbox[2], car_bbox[3]),
                                                               car_score,
                                                               '[{} {} {} {}]'.format(
                                                                   license_plate_bbox[0], license_plate_bbox[1],
                                                                   license_plate_bbox[2], license_plate_bbox[3]),
                                                               license_plate_bbox_score,
                                                               license_number,
                                                               license_number_score))
    f.close()


def license_complies_format(text):
    # """
    # Check if the license plate text complies with the required format.

    # Args:
    #     text (str): License plate text.

    # Returns:
    #     bool: True if the license plate complies with the format, False otherwise.
    # """
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
    # """
    # Format the license plate text by converting characters using the mapping dictionaries.

    # Args:
    #     text (str): License plate text.

    # Returns:
    #     str: Formatted license plate text.
    # """
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
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, score, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            print('abheek')
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1, -1
