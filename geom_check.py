"""Geometrical verification check"""

import cv2
import logging
import numpy as np
import os
import parameters
from processing_images import *


def find_homography(match_img, query_img, extractor, matcher, ransac_thres=15):
    """Find homography with RANSAC algorithm between source and destination points
    for query and match image features """
    mask = np.empty([1, 1])
    good = []
    if match_img is not None:
        if query_img is not None:
            kp1, des1 = extractor.detectAndCompute(match_img, None)
            kp2, des2 = extractor.detectAndCompute(query_img, None)
            matches = matcher.knnMatch(des1, des2, k=2)  # Match
            # Store all good matches as per Lowe's ratio test
            if len(matches) > 0:
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good.append(m)
                if len(
                        good) > parameters.MIN_MATCH_COUNT:  # condition for min number of matches
                    src_pts = np.float32(
                        [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    # Find homography between source and destination points
                    H, mask = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, ransac_thres)
    return mask, good


def ransac_test_onmatch(match_img, query_img, contrast=0.04, edge=10, ransac_thres=15):
    """Return number of inliers for given match and query images"""
    extractor = cv2.xfeatures2d.SIFT_create(
        nfeatures=parameters.FEATURES_CLUSTERS,
        contrastThreshold=contrast,
        edgeThreshold=edge)
    matcher = cv2.BFMatcher()
    inliers = 0
    mask, _ = find_homography(match_img, query_img,
                              extractor, matcher, ransac_thres)
    if mask is not None:
        matchesMask = mask.ravel().tolist()
        if matchesMask is not None:
            inliers = sum(1 for i in matchesMask if i == 1)
    return inliers


def show_inliers(match_img, query_img, contrast=0.04, edge=10, ransac_thres=15):
    extractor = cv2.xfeatures2d.SIFT_create(
        nfeatures=parameters.FEATURES_CLUSTERS,
        contrastThreshold=contrast,
        edgeThreshold=edge)
    M, good = find_homography(match_img, query_img, ransac_thres)
    # extract features from match image
    kp1, _ = extractor.detectAndCompute(match_img, None)
    # extract features from query image
    kp2, _ = extractor.detectAndCompute(query_img, None)
    h, w = query_img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    if np.all(abs(dst[0] - dst[1]) > 1):
        # Draw a box around found object
        img_new = cv2.polylines(query_img, [np.int32(dst)], True, 255,3, cv2.LINE_AA)
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=2)
        img3 = cv2.drawMatches(
            match_img,
            kp1,
            query_img,
            kp2,
            good,
            None,
            **draw_params)
        ndir3 = os.path.join(parameters.RESULTS_DIR, 'show_inliers.jpg')
        cv2.imwrite(ndir3, img3)
