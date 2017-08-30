"""Functions to build image retrieval engine """

import csv
import logging.config
import operator
import os
import pickle
import time
from PIL import Image
import vse
import yaml

import parameters
from detect_objects import run_yolo_onpic, crop_box_for_class
from processing_images import read_image
from cnn_feature_extraction import VisualSearchEngine_cnn


def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def timeit(func):
    """Measure function running time"""
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        print('%s was completed in %2.2f sec' %
              (func.__name__, te - ts))
        return result
    return timed


def read_ground_truth(filename, opposite=False):
    """Read the ground truth data file"""
    ground_truth = {}
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if not opposite:
                # ground truth[scene] = product
                ground_truth[row[0]] = row[1]
            else:
                # read vice versa: ground truth[product] = scene
                ground_truth[row[1]] = row[0]
    return ground_truth


def load(filename):
    """Loads data from file using pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def cnn_descriptor(
        directory,
        features_clusters):
    ranker = vse.SimpleRanker(hist_comparator=vse.Intersection())
    index = vse.InvertedIndex(ranker=ranker,
                              recognized_visual_words=features_clusters)
    vse_engine = VisualSearchEngine_cnn(index)
    for filename in os.listdir(directory):
        ndir = os.path.join(directory, filename)
        try:
            image = Image.open(filename)
            vse_engine.add_to_index_cnn(filename, image, ndir)
        except OSError:
            print('Not an image', filename)
            continue
    return vse_engine


def geom_check(query_img, result, results_dir, nb_best):
    """
    Verify matches geometrically and return nb_best of matches from the initial result.
    """
    if nb_best > parameters.NB_MATCHES:
        return result  # more best matches to be returned than are given
    match_inliers = {}
    for match in result:
        match_img = read_image(match[0], results_dir)
        inliers = ransac_test_onmatch(query_img, match_img)
        if inliers > 0:
            match_inliers[match[0]] = inliers
    # if we don't have enough matches, add the best matches from initial result
    if len(match_inliers) < nb_best:
        nb_to_be_added = min(nb_best - len(match_inliers), len(result))
        for i in range(0, nb_to_be_added):
            if result[i][0] in match_inliers:
                i += 1
                nb_to_be_added += 1
            match_inliers[result[i][0]] = 0
    # Return nb_best of the matches with highest nb of inliers
    best_matches = dict(sorted(match_inliers.items(),
                               key=operator.itemgetter(1), reverse=True)[:nb_best])
    return best_matches


def evaluation_test(query, result, ground_truth):
    """
    Check if any of the returned results are correct.
    """
    correct = 0
    for match in result:
        if isinstance(match, str):
            matching_name = match
        else:
            matching_name = match[0]
        if query in ground_truth:
            if ground_truth[query] == matching_name:
                correct = 1
    return correct


def test_on_set(test_set, results_set, engine, ground_truth):
    """For each image in test_set evaluate if any of the n guesses is correct.
        Returns value between 0 and 1"""
    img_count = 0
    correct_count = 0
    ground_truth_data = read_ground_truth(ground_truth)
    for filename in os.listdir(test_set):
        ndir = os.path.join(test_set, filename)
        try:
            test_image = Image.open(ndir)
            img_count += 1
            _, _, boxes = run_detector_onpic(ndir)
            crop_clock = crop_box_for_class(boxes, ndir, BOXES_DIR, 'clock')
            best_result = return_similar(crop_clock, results_set, engine)
            eval1 = evaluation_test(
                filename, best_result, ground_truth_data)
            # update sum of corrects with 0 or 1
            correct_count += eval1
            if eval1 == 1:
                print('Correct guess for %s' % filename)
                logging.debug('Correct guess for %s' % filename)
            accuracy = (correct_count / img_count) * 100.0
            print('Accuracy so far is %f ' % accuracy)
        except OSError:
            print('Not an image', filename)
            continue
    logging.info('Result on %s is  %f' % (test_set, correct_count / img_count))
    print(correct_count / img_count)
    return correct_count / img_count


def initiate_engine(results_dir, feature_model, vocabulary=None):
    if feature_model == "bovw":
        print('BOVW model not supported in web app!')
    else:
        if feature_model == "resnet":
            features_dim = 2048
        else:
            features_dim = 4096
        vse_engine = cnn_descriptor(
            results_dir,
            features_dim)
        file_name = os.path.join(
            'pickles/', os.path.basename(results_dir) + '_' + feature_model + '.pickle')
        fileObject = open(file_name, 'wb')
        pickle.dump(vse_engine, fileObject)
        fileObject.close()
    return vse_engine


def return_similar(filename, results_dir, engine, nb_matches=parameters.NB_MATCHES, geom_check=False, nb_best=parameters.NB_BEST):
    """Main function for returning similar images"""
    print('Return_similar for', filename)
    try:
        test_image = Image.open(filename)
        if parameters.FEATURE_MODEL == "bovw":
            result = engine.find_similar(test_image, nb_matches)
        else:
            result = engine.find_similar(filename, nb_matches)
        # print('result is ', result)
        if geom_check:
            print('Geom check is not supported in web app!')
        else:
            best_result = {}
            for entry in result:
                best_result[entry[0]] = 0
            # print('best result', best_result)
            return best_result
    except OSError:
        print('Wrong image file!', filename)
        return ('not_found.jpg', 0)
