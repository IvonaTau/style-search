import argparse
import keras.applications.resnet50
import numpy as np
import os
import pickle
import time
import vse

from keras.preprocessing import image
from keras.models import Model, load_model

from detect_objects import crop_box_for_class, detect_class_onpic
import parameters

model_extension = parameters.FEATURE_MODEL

if model_extension == "vgg19":
    base_model = keras.applications.vgg19.VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer(
        parameters.CNN_layer).output)
elif model_extension == "vgg16":
    base_model = keras.applications.vgg16.VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer(
        parameters.CNN_layer).output)
elif model_extension == "resnet":
    model = keras.applications.resnet50.ResNet50(include_top=False)
else:
    print('Not a CNN feature extraction model')


class VisualSearchEngine_cnn:

    def __init__(self, image_index):
        self.image_index = image_index

    def add_to_index_cnn(self, image_id, image, image_path):
        """Adds CNN features to image"""
        print('Adding %s to engine' % image_path)
        features = extract_features_cnn(image_path)
        self.image_index[image_id] = features

    def remove_from_index(self, image_id):
        """Removes item with image_id."""
        del self.image_index[image_id]

    def find_similar(self, image_path, n=1):
        """Returns at most n similar images."""
        try:
            with open("pickles/gallery_cnn_features_" + model_extension + ".pickle", "rb") as handle:
                gallery_cnn_features = pickle.load(handle)
                query_features = gallery_cnn_features[
                    os.path.basename(image_path)]
        except:
            print('No features for this image found, extracting from CNN')
            query_features = extract_features_cnn(image_path)

        return self.image_index.find(query_features, n)


def create_vse(features_number=4096):
    """Create visual search engine with default configuration."""
    ranker = vse.SimpleRanker(hist_comparator=vse.Intersection())
    inverted_index = vse.InvertedIndex(
        ranker=ranker, recognized_visual_words=features_number)
    return VisualSearchEngine_cnn(inverted_index)


def extract_features_cnn(img_path):
    """Returns a normalized features vector for image path and model specified in parameters file """
    print('Using model', parameters.FEATURE_MODEL)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if model_extension == "vgg19":
        x = keras.applications.vgg19.preprocess_input(x)
    elif model_extension == "vgg16":
        x = keras.applications.vgg16.preprocess_input(x)
    elif model_extension == "resnet":
        x = keras.applications.resnet50.preprocess_input(x)
    else:
        print('Wrong model name')
    model_features = model.predict(x, batch_size=1)
    total_sum = sum(model_features[0])
    features_norm = np.array(
        [val / total_sum for val in model_features[0]], dtype=np.float32)
    if parameters.FEATURE_MODEL == "resnet":
        print("reshaping resnet")
        features_norm = features_norm.reshape(2048)
    return features_norm


def save_image_features(img_path, features_file='pickles/gallery_cnn_features_' + model_extension + '.pickle'):
    """Saves image features vector to pickle file """
    try:
        with open(features_file, 'rb') as handle:
            features_set = pickle.load(handle)
    except FileNotFoundError:
        features_set = {}
    image_name = os.path.basename(img_path)
    if image_name in features_set:
        print(image_name, 'features already in the file ! ')
        return features_set[image_name]
    else:
        print('Extracting features for ', image_name)
        query_features = extract_features_cnn(img_path)
        features_set[image_name] = query_features
        fileObject = open(features_file, 'wb')
        pickle.dump(features_set, fileObject)
        fileObject.close()
        return query_features


def save_features_for_objects(images_path, bounding_boxes_dir='./app/static/bounding_boxes',
                              detections_file='pickles/bounding_boxes.pickle'):
    """Detects objects , extracts features for objects and saves features to pickle file"""
    for image_name in os.listdir(images_path):
        bound_boxes = detect_objects(os.path.join(images_path, image_name))
        if type(bound_boxes) == int:
            print(bound_boxes)
            print('BOUND BOXES NOT FOUND!')
        else:
            object_class, _ = detect_class_onpic(
                bound_boxes, parameters.ALLOWED_CLASSES)
            object_image = crop_box_for_class(bound_boxes, os.path.join(
                images_path, image_name), bounding_boxes_dir, object_class)
            save_image_features(object_image)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory',
        help="Directory of images to be processed for CNN feature extraction")
    args = parser.parse_args()

    save_features_for_objects(args.directory)
