"""Similar images finder based on visual words vocabulary"""

import argparse
import csv
import cv2
import logging.config
import os
import vse

import parameters  # model parameters defined in a separate file
from processing_images import process_all_images
from finder import read_ground_truth, test_on_set, userprompt_finder,  setup_logging, bow_descriptor, cnn_descriptor


if __name__ == '__main__':
    setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_directory",
        help="Directory for images used as search query")
    parser.add_argument(
        "results_directory",
        help="Directory for results images")
    parser.add_argument(
        "ground_truth",
        help="File with relationships for objects from query directory and results directory in csv format")
    parser.add_argument(
        "voc_file",
        help="Directory for vocabulary file in yml format")
    args = parser.parse_args()

    # Read the ground truth data file indicating object to scene relations.
    ground_truth = read_ground_truth(args.ground_truth)

    # Load the vocabulary from disc (should be created with
    # create_visual_vocabulary.py)
    vocabulary = vse.load(args.voc_file)

    # Initialize features extractor (SIFT) and features matcher (Brute Force)
    EXTRACTOR = cv2.xfeatures2d.SIFT_create(
        nfeatures=parameters.FEATURES_CLUSTERS,
        contrastThreshold=parameters.CONTRAST_THRES,
        edgeThreshold=parameters.EDGE_THRES)
    MATCHER = cv2.BFMatcher()

    # Create visual search engine based on visual words vocabulary and set of
    # your query images
    vse_engine = cnn_descriptor(
        args.results_directory,
        vocabulary,
        EXTRACTOR,
        MATCHER,
        parameters.FEATURES_CLUSTERS)

    # For all images in the directory find user defined number of closest matches and check
    # if they were identified correctly by comparing to ground truth.
    test_on_set(
        args.query_directory,
        args.results_directory,
        vse_engine,
        args.ground_truth)

    # User prompt for a specific file to be searched
    # userprompt_finder(args.results_directory,
    #                   args.query_directory, vocabulary, args.ground_truth)
