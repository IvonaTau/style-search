"""This file defines global model parameters"""

#Parameters for feature extractor and matcher
CONTRAST_THRES = 0.05
EDGE_THRES = 11
NORM_TYPE = 'cv2.NORM_L2'

#Number of visual words features
FEATURES_CLUSTERS = 1000
#Number of returned matches in query
NB_MATCHES = 6
NB_BEST = 6 #for RANSAC geom verification
#Threshold for verification
BEST_THRES = 70
GEOM_THRES = 30

#SIFT parameters
MIN_MATCH_COUNT = 5
MIN_DIST = 0.7

#Yolo
YOLO_THRES = 0.2
# YOLO_THRES = 0.01
YOLO = True #only for accuracy calculations

ALLOWED_CLASSES = ['diningtable', 'chair', 'sofa', 'pottedplant', 'table', 'clock', 'bed', 'plant_pot']
ACCURACY_CLASSES = ['diningtable', 'chair', 'sofa', 'pottedplant']#use diningtable, pottedplant

#Feature extraction model in [vgg16, vgg19, resnet, bovw]
FEATURE_MODEL='resnet'
CNN_layer = 'fc2'