# README #

### What is this repository for? ###

* Visual/textual search blend demo version

### Dependencies ###

* Installation of Open CV 2
* Python 3

### Web app ###
* Start on localhost by running run.py

* Configuration of Flask app interface: app/web_interface.py


### Notebooks ###

* Results for IKEA Dataset - contains accuracy calculations for visual search, recall curve on IKEA dataset
* Interior style dataset benchmark - contains accuracy calculations for visual and textual search on Style dataset
* Results for calculating similarity - contains similarity metric calculations for different text queries and objects in IKEA dataset

### Visual Search ###

* Visual search functions: finder.py
* Visual feature extraction: cnn_feature_extraction.py
* Functions for YOLO object detection: detect_objects.py
* Model parameters: parameters.py

### Textual Search ###
* Query transformation using SVD and finding n-nearest neigbhours: search_engine.py
* Word2vec and Countvect "training": training.py
* tSNE visualization: embedding.py
* blender.py - leftover
* Query transformation using LSTM is in the jupyter notebook sent on style-search channel on slack