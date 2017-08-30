import argparse
import ctypes
import os
import pickle
from PIL import Image
from shutil import copy2

import parameters

mylib = ctypes.cdll.LoadLibrary('./libdarknetlnx.so')

# relevant structures from C
class image (ctypes.Structure):
    _fields_ = [('h', ctypes.c_int),
                ('w', ctypes.c_int),
                ('c', ctypes.c_int),
                ('data', ctypes.POINTER(ctypes.c_float))]

class box (ctypes.Structure):
    _fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float),
                ('w', ctypes.c_float),
                ('h', ctypes.c_float), ]


# C funtions bindings

# void test_detector(char *datacfg, char *cfgfile, char *weightfile, char
# *filename, float thresh, float hier_thresh, char *outfile)
_test_detector = mylib.test_detector
_test_detector.argtypes = (ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_char_p)
_test_detector.restype = None

def test_detector(datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, outfile):
    """Binding for YOLO 9000 function which runs testing detector on image"""
    _test_detector(datacfg, cfgfile, weightfile,
                   filename, thresh, hier_thresh, outfile)


def read_bounding_boxes(filename):
    """Reads bounding boxes from text file. Returns weight, height and a list of objects that were detected in the picture"""
    f = open(filename)
    objects = []
    weight = 0
    height = 0
    for line in f:
        print(line)
        first_word = line.split(';')[0]
        if first_word == "Dimensions":
            weight = line.split(';')[1]
            height = line.split(';')[2]
        if first_word == "Object":
            objects.append((line.split(';')[1], line.split(';')[2], line.split(';')[4],
                            line.split(';')[5], line.split(';')[6], line.split(';')[7]))
    return weight, height, objects


def crop_bounding_box_from_image(bounding_box, image_path, with_margin=True):
    """Returns cropped bounding box from image"""
    original_image = Image.open(image_path)
    margin_length = 0
    margin_height = 0
    if with_margin:
        # Extend bounding box length and height by 20%
        margin_length = 0.1 * (int(bounding_box[3]) - int(bounding_box[2]))
        margin_height = 0.1 * (int(bounding_box[5]) - int(bounding_box[4]))
    cropped_image = original_image.crop((int(bounding_box[2]) - margin_length, int(bounding_box[4]) - margin_height,
                                         int(bounding_box[3]) + margin_length, int(bounding_box[5]) + margin_height))
    return cropped_image


def crop_all_bounding_boxes(boxes, image_path, crop_path):
    """For list of bounding boxes crops all detected objects from source image and saves in crop_path """
    index = 0
    for box in boxes:
        object_class = box[0]
        cropped_image = crop_bounding_box_from_image(
            box, image_path, crop_path)
        filename = object_class + "_" + os.path.basename(image_path)
        while os.path.isfile(os.path.join(crop_path, filename)):
            print('File %s already exists!' % (filename))
            index += 1
            filename = str(index) + "_" + filename
        cropped_image.save(filename)


def crop_box_for_class(boxes, image_path, crop_path, object_class):
    """Crop image only for bounding box with highest probability in given object class"""
    if object_class == "all":
        return image_path
    class_prob = 0
    best_box = []
    for box in boxes:
        box_prob = float(box[1].strip('%')) / 100.0
        if box[0] == object_class:
            if box_prob > class_prob:
                class_prob = box_prob
                best_box = box
    cropped_image = crop_bounding_box_from_image(
        best_box, image_path, with_margin=True)
    width, height = cropped_image.size
    if (height > 140) and (width > 140):
        cropped_image_path = object_class + "_" + os.path.basename(image_path)
        cropped_image.save(os.path.join(crop_path, cropped_image_path))
        # image to show / no margin
        cropped_image_show = crop_bounding_box_from_image(
            best_box, image_path, with_margin=False)
        cropped_image_show_path = object_class + \
            "_show_" + os.path.basename(image_path)
        cropped_image_show.save(os.path.join(
            crop_path, cropped_image_show_path))
        return os.path.join(crop_path, cropped_image_path)
    else:
        return image_path


def detect_class_onpic(boxes, allowed_classes):
    """For a list of bounding boxes return a class from allowed classes that has the highest probability"""
    object_class = "all"
    highest_prob = 0
    for box in boxes:
        box_prob = float(box[1].strip('%')) / 100.0
        if box[0] in allowed_classes and box_prob > highest_prob:
            highest_prob = box_prob
            object_class = box[0]
    return object_class, highest_prob


def run_yolo_onpic(image_path):
    """For image path run YOLO object detection and return list of objects that were detected along with weight and height """
    try:
        Image.open(image_path)
        # print('running detector on %s' %  image_path)
    except:
        print('Cannot open image', image_path)
        return 0, 0, 0
    output_file = "predictions_" + os.path.basename(image_path)
    test_detector(b'cfg/coco.data', b'cfg/yolo.cfg', b'yolo.weights',
                  image_path.encode('utf-8'), parameters.YOLO_THRES, 0.5, output_file.encode('utf-8'))
    w, h, o = read_bounding_boxes('bounding_boxes.txt')
    return w, h, o


def run_yolo_indir(images_path):
    """For every image in images_path run YOLO object detection and crop detected objects"""
    for filename in os.listdir(images_path):
        try:
            # print(filename)
            Image.open(os.path.join(images_path, filename))
            test_detector(b'cfg/voc.data', b'cfg/yolo.cfg', b'yolo.weights', os.path.join(
                images_path, filename).encode('utf-8'), parameters.YOLO_THRES, 0.5)
            w, h, o = read_bounding_boxes('bounding_boxes.txt')
            crop_all_bounding_boxes(o, filename, os.path.join, images_path)
        except:
            print('Cannot test image', filename)
            continue


def detect_objects_on_image(image_path, detections_file='pickles/bounding_boxes.pickle'):
    """For image path return a list of detected bounding boxes"""
    image_name = os.path.basename(image_path)
    try:
        with open(detections_file, 'rb') as handle:
            detections = pickle.load(handle)
    except FileNotFoundError:
        print('Detections file not found!')
        detections = {}
    if image_name in detections:
        print(image_name, 'is already in detections file!')
        print('Bounding boxes from file', detections[image_name])
        return detections[image_name]
    else:
        print('Adding to detections file', image_name)
        _, _, bound_boxes = run_yolo_onpic(image_path)
        detections[image_name] = bound_boxes
        print('Bounding boxes', bound_boxes)
        fileObject = open(detections_file, 'wb')
        pickle.dump(detections, fileObject)
        fileObject.close()
        return bound_boxes


def initiate_yolo_detect(images_path, save_to_path, detections_file='pickles/bounding_boxes.pickle'):
    """detect objects for all images in path and save the bounding boxes to pickle path """
    for filename in os.listdir(images_path):
        bound_boxes = detect_objects_on_image(
            os.path.join(images_path, filename), detections_file)
        predictions_path = os.path.join(
            save_to_path, 'predictions_' + filename)
        print('predictions path', predictions_path)
        copy2('predictions_' + os.path.basename(image_directory) +
              '.png', predictions_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory',
        help="Directory of images to be processed for object detection")
    parser.add_argument('save_to_path')
    # parser.add_argument('detections_file')
    args = parser.parse_args()
    # Build visual vocabulary
    initiate_yolo_detect(args.directory, args.save_to_path)
