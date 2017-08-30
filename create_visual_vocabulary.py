"""Processes images from the selected directory and creates vocabulary of visual words"""

import argparse
import logging
import os
import time
import cv2
import vse

import parameters  # model parameters defined in a separate file
from processing_images import read_image, process_all_images
from finder import setup_logging, timeit


@timeit
def create_vocabulary(directory, features_clusters, extractor):
    """Reads all images in the directory, extracts features.
       Returns vocabulary of n features"""
    descriptors = []
    for filename in os.listdir(directory):
        img = read_image(filename, directory)
        if img is not None:
            _, des = extractor.detectAndCompute(img, None)
            if des is not None:
                logging.debug('%s has number of decriptors: %d' %
                              (filename, len(des)))
                descriptors.append(des)
    # Initiate bag of words trainer
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(features_clusters)
    for desc in descriptors:
        bow_kmeans_trainer.add(desc)
    logging.info("K-means clustering for features in progress ...")
    vocabulary = bow_kmeans_trainer.cluster()
    # Save vocabulary data to file using pickle
    ndir = os.path.join(directory, 'vocabulary.yml')
    vse.save(ndir, vocabulary)
    return vocabulary


def visual_vocabulary(voc_directory):
    """Read all images, process them to new directory (black & white, resize)
        and create vocabulary"""
    extractor = cv2.xfeatures2d.SIFT_create(
        nfeatures=parameters.FEATURES_CLUSTERS,
        contrastThreshold=parameters.CONTRAST_THRES,
        edgeThreshold=parameters.EDGE_THRES)  # initialize feature extractor

    process_all_images(voc_directory)  # preprocess all images
    processed_directory = os.path.join(voc_directory, 'processed')
    create_vocabulary(
        processed_directory,
        parameters.FEATURES_CLUSTERS,
        extractor)
    logging.info('Vocabulary was successfully created!')

if __name__ == '__main__':
    setup_logging()
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory',
        help="Directory of images to be processed for vocabulary creation")
    args = parser.parse_args()
    # Build visual vocabulary
    visual_vocabulary(args.directory)
