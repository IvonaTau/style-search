"""Processing images to extract features

This module reads the image data, preprocesses images (resize, convert to B&W).
Then keypoints and descriptors are extracted from images
Producing Bag of Visual Words vocabulary by training the bags
with the clustered feature descriptors.

"""

import logging
import os
import cv2
import vse


def read_image(filename, directory):
    """Reads image from the given directory"""
    if filename is not None:
        ndir = os.path.join(directory, filename)
        image = cv2.imread(ndir, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
        else:
            logging.debug('No such image %s in given directory' % (filename))


def process_image(filename, directory):
    """Reads image to black and white,
       resizes image where MAX_SIZE  = 1000, MIN_SIZE = 150"""
    image = read_image(filename, directory)
    if image is not None:
        height, width = image.shape[:2]
        if min(height, width) < 150:
            logging.debug(filename, " is smaller than 150px")
        img_conv = vse.convert_image(image)
        return img_conv


def process_all_images(directory):
    """Processes all images and saves them to new directory"""
    print('Processing all images from the diretory: ', directory)
    new_directory = os.path.join(directory, 'processed')
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    for filename in os.listdir(directory):
        img = process_image(filename, directory)
        if img is not None:
            ndir = os.path.join(new_directory, filename)
            cv2.imwrite(ndir, img)


def draw_keypoints(directory, extractor):
    """ Draws keypoints for images in given directory and save it to new one"""
    new_directory = os.path.join(directory, 'processed')
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    for filename in os.listdir(directory):
        image = process_image(filename, directory)
        if image is not None:
            print('Drawing keypoints for: ', filename)
            key_points, _ = extractor.detectAndCompute(image, None)
            img3 = cv2.drawKeypoints(
                image, key_points, image,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            ndir = os.path.join(new_directory, filename)
            cv2.imwrite(ndir, img3)
