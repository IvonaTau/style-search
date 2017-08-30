from flask import Flask
import os, os.path
import gensim
import pickle
import keras.applications.vgg19 
import keras.applications.vgg16
import keras.applications.resnet50
from keras.models import Model
import time
 
from finder import initiate_engine, load
from training import CountVectModel
from search_engine import SearchEngine
import parameters
 

app = Flask(__name__)
app.secret_key = os.urandom(12)
app.config.from_object('config')
app.static_folder = 'static'
 
app.config['UPLOAD_FOLDER'] = './app/static/uploads'
app.config['BOUNDING_BOXES']='./app/static/bounding_boxes'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['YOLO_FOLDER'] = './app/static/yolo_detections'
 
app.config['CLOCK_DIR'] = './app/static/images/clock'
app.config['CHAIR_DIR'] = './app/static/images/chair'
app.config['POT_DIR'] = './app/static/images/plant_pot'
app.config['SOFA_DIR'] = './app/static/images/sofa'
app.config['TABLE_DIR'] = './app/static/images/table'
app.config['ROOM_DIR'] = './app/static/images/room_scenes'
app.config['BED_DIR'] = './app/static/images/bed'
 
app.config['VOC_CLOCK'] = './app/static/images/clock/processed/vocabulary300_clock_lamem.yml'
app.config['VOC_CHAIR'] = './app/static/images/chair/processed/vocabulary.yml'
app.config['VOC_SOFA'] = './app/static/images/sofa/processed/vocabulary.yml'
app.config['VOC_TABLE'] = './app/static/images/table/processed/vocabulary.yml'
app.config['VOC_BED'] = './app/static/images/bed/processed/vocabulary.yml'
 
app.config['ALLOWED_CLASSES'] = ['diningtable', 'chair', 'sofa', 'pottedplant', 'table', 'clock', 'bed']

app.config['CLASS_COLORS'] = {"chair": "#1a8cff", "pottedplant":"#ff9933", "sofa":"#ff0066", "clock":"#00ff99", "diningtable":"#66ffcc",
    "bed":"#00ff00", "room":"#000000"}

app.config['OBJECT_FEATURES_FILE'] = 'pickles/object_features.pickle'
 

# load dict
with open('pickles/products_dict.p', 'rb') as handle:
    products_dict = pickle.load(handle)
 
# load w2vec model
with open('pickles/word2vec_model.p', 'rb') as f:
    model = pickle.load(f)


# build search engine
vectorizer = CountVectModel(products_dict)
search_engine = SearchEngine(products_dict, vectorizer, model)

#load yolo detections
with open('pickles/bounding_boxes.pickle', 'rb') as f:
    YOLO_detections = pickle.load(f)
    print('Gallery yolo detections loaded!')
 
model_extension = parameters.FEATURE_MODEL


#load cnn features
try:
    with open('pickles/gallery_cnn_features_' + model_extension + '.pickle', 'rb') as f:
        GALLERY_FEATURES = pickle.load(f)
        print('Gallery CNN features loaded!')
except FileNotFoundError:
    print('No CNN features for model', model_extension)


print('Model extension is', model_extension)

#create model
if model_extension == "vgg19":
    print('VGG19 not supported in web app!')
elif model_extension == "vgg16":
    print('VGG16 not supported in web app!')
elif model_extension == "resnet":
    model = keras.applications.resnet50.ResNet50(include_top=False)
else:
    print('Wrong model in parameters')
 

try:
    with open('pickles/clock_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_CLOCK = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_CLOCK = initiate_engine(app.config['CLOCK_DIR'], model_extension, load(app.config['VOC_CLOCK']))
try:
    with open('pickles/bed_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_BED = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_BED = initiate_engine(app.config['BED_DIR'], model_extension, load(app.config['VOC_BED']))
try:
    with open('pickles/chair_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_CHAIR = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_CHAIR = initiate_engine(app.config['CHAIR_DIR'], model_extension, load(app.config['VOC_CHAIR']))
try:
    with open('pickles/plant_pot_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_POT = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_POT = initiate_engine(app.config['POT_DIR'], model_extension, load(app.config['VOC_CLOCK']))
try:
    with open('pickles/sofa_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_SOFA = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_SOFA = initiate_engine(app.config['SOFA_DIR'], model_extension, load(app.config['VOC_SOFA']))
try:
    with open('pickles/table_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_TABLE = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    ENGINE_TABLE = initiate_engine(app.config['TABLE_DIR'], model_extension, load(app.config['VOC_TABLE']))

print('All engines initiated')
 
 
from app import web_interface