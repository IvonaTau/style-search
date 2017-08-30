"""Web application interface

"""

import os
from flask import Flask, render_template, request, flash, session
import numpy as np
import pickle
from PIL import Image
from shutil import copy2
from werkzeug.utils import secure_filename

from app import app, search_engine, ENGINE_CHAIR, ENGINE_CLOCK, ENGINE_SOFA, ENGINE_BED, ENGINE_POT, ENGINE_TABLE
from finder import return_similar
from detect_objects import detect_class_onpic, crop_box_for_class, detect_objects_on_image, crop_bounding_box_from_image
from cnn_feature_extraction import extract_features_cnn, save_image_features



def get_engine(image_directory, allowed_classes):
    bound_boxes = detect_objects_on_image(image_directory)
    predictions_path = os.path.join(
        app.config['YOLO_FOLDER'], 'predictions_' + os.path.basename(image_directory))
    try:
        copy2('predictions_' + os.path.basename(image_directory) +
              '.png', predictions_path)
        os.remove('predictions_' + os.path.basename(image_directory) + '.png')
    except FileNotFoundError:
        print('Nothing to copy - no yolo predictions image found')
    object_class, _ = detect_class_onpic(bound_boxes, allowed_classes)
    print('Detected object class from bounding boxes', object_class)
    search_dir, engine, static_path = get_directories(object_class)
    return search_dir, engine, static_path, object_class, bound_boxes


def get_directories(object_class):
    print('Getting directories for ', object_class)
    if object_class == "chair":
        search_dir = app.config['CHAIR_DIR']
        engine = ENGINE_CHAIR
        static_path = '/static/images/chair'
    elif object_class == "clock":
        search_dir = app.config['CLOCK_DIR']
        engine = ENGINE_CLOCK
        static_path = '/static/images/clock'
    elif object_class == "pottedplant":
        search_dir = app.config['POT_DIR']
        engine = ENGINE_POT
        static_path = '/static/images/plant_pot'
    elif object_class == "sofa":
        search_dir = app.config['SOFA_DIR']
        engine = ENGINE_SOFA
        static_path = '/static/images/sofa'
    elif object_class in ("diningtable", "table"):
        search_dir = app.config['TABLE_DIR']
        engine = ENGINE_TABLE
        static_path = '/static/images/table'
    elif object_class == "bed":
        search_dir = app.config['BED_DIR']
        engine = ENGINE_BED
        static_path = '/static/images/bed'
    else:
        search_dir = app.config['CLOCK_DIR']
        engine = ENGINE_CLOCK
        static_path = '/static/images/clock'
    print('Searching for results in ', search_dir)
    return search_dir, engine, static_path


def get_clicked_object(base_image_path, bound_boxes, image_x, image_y, width):
    with Image.open(base_image_path) as img:
        img_width, _ = img.size
    # rescale x and y coordinates accordingly to
    rescale_factor = int(img_width) / int(width)
    image_x = float(image_x) * rescale_factor
    image_y = float(image_y) * rescale_factor
    print('Different image dimensions. Rescaling image by', rescale_factor)
    for box in bound_boxes:
        if box[0] not in app.config['ALLOWED_CLASSES']:
            continue
        else:
            x1 = int(box[2])
            x2 = int(box[3])
            y1 = int(box[4])
            y2 = int(box[5])
            if x1 < image_x and image_x < x2:
                if y1 < image_y and image_y < y2:
                    print('Found a bounding box!', box)
                    return box
    print('No valid box was clicked')
    return 0


def return_feature_blender_results(initial_image, w2vec_results, countvect_results):
    initial_query = w2vec_results + countvect_results
    print('initial query', initial_query)
    # initial_image = './app' + visual_results[0]
    print('nitial_image', initial_image)
    distances = {}
    initial_features = save_image_features(initial_image, app.config['OBJECT_FEATURES_FILE'])
    for product_path in initial_query:
        image_features = save_image_features(product_path, app.config['OBJECT_FEATURES_FILE'])
        distances[product_path] = np.linalg.norm(
            image_features - initial_features)
    sorted_array = [key for key in sorted(
        distances, key=distances.__getitem__)]
    return sorted_array[:8]


@app.route('/')
@app.route('/index')
def index():
    """Render template for homepage"""
    images = []
    for filename in os.listdir('./app/static/images/room_scenes'):
        if not filename.endswith('.jpg'):
            continue
        im = Image.open(os.path.join(
            './app/static/images/room_scenes', filename))
        w, h = im.size
        images.append({
            'width': int(w),
            'height': int(h),
            'src': os.path.join('./static/images/room_scenes', filename)
        })
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('scene_gallery.html', images=images)


@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'tooploox' and request.form['username'] == 'admin':
        session['logged_in'] = True
    elif request.form['password'] == 'fedcsis17' and request.form['username'] == 'midi':
        session['logged_in'] = True
    else:
        flash('Wrong password!')
    return index()


@app.route('/about')
def about_section():
    """Render template for about section"""
    return render_template('about.html')


@app.route('/upload_image')
def upload_file():
    """Render template for uploading your own image page"""
    return render_template('upload_image.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    """Get the image uploaded by user and search for similar"""
    # load img_dict
    with open('img_to_text.p', 'rb') as handle:
        img_to_text = pickle.load(handle)

    if request.method == 'POST':
        f = request.files['file']

        # crop functionality
        # x1 = int(request.form.get('x1'))
        # y1 = int(request.form.get('y1'))
        # x2 = int(request.form.get('x2'))
        # y2 = int(request.form.get('y2'))

        image_filename = secure_filename(f.filename)
        query = request.form['query']
        #query = img_to_text['images/' + image_filename]
        image_directory = os.path.join(
            app.config['UPLOAD_FOLDER'], image_filename)
        f.save(image_directory)

        # original_image = Image.open(image_directory)
        # cropped_image = original_image.crop((x1, y1, x2, y2))
        # crop_filename = 'cropped_' + image_filename
        # cropped_image.save(os.path.join(
        #     app.config['UPLOAD_FOLDER'], crop_filename))

        search_dir, engine, static_path, object_class, bound_boxes = get_engine(
            image_directory, app.config['ALLOWED_CLASSES'])
        print('Object class is', object_class)
        predictions_path = os.path.join(
            app.config['YOLO_FOLDER'], 'predictions_' + os.path.basename(image_directory))

        if object_class == "all":
            object_image = image_directory
            object_image_path = os.path.join('static/uploads', image_filename)
            return render_template('furniture_not_found.html',
                                   query_image=os.path.join('/static/yolo_detections/', os.path.basename(predictions_path)))
        else:
            object_image = crop_box_for_class(bound_boxes, image_directory, app.config[
                                              'BOUNDING_BOXES'], object_class)
            object_image_path = os.path.join(
                '/static/bounding_boxes', object_class + "_show_" + os.path.basename(image_directory))
            object_text = object_class

        similar_images = return_similar(object_image, search_dir, engine)
        print('text query', query)

        if query == '':
            if object_class == 'pottedplant':
                images_countvect = search_engine.process_query('plant pot')
                images_w2vec = search_engine.process_query_w2vec('plant pot')
            else:
                images_countvect = search_engine.process_query(object_class)
                images_w2vec = search_engine.process_query_w2vec(object_class)
        else:
            images_countvect = search_engine.process_query(query)
            images_w2vec = search_engine.process_query_w2vec(query)

        result_images_w2vec = [os.path.join(
            '/static', image) for image in images_w2vec]
        result_images_countvect = [os.path.join(
            '/static', image) for image in images_countvect]

        result_images = [os.path.join(static_path, image)
                         for image in similar_images]
        top4_blend = result_images[
            :4] + result_images_w2vec[:4] + result_images_countvect[:4]

        return render_template('pre_search_image.html',
                               title="Search for similar image",
                               query_image=os.path.join(
                                   '/static/yolo_detections/', os.path.basename(predictions_path)),
                               text_query=query,
                               object_image=object_image_path,
                               object_class=object_text,
                               result_images=result_images,
                               result_images_w2vec=result_images_w2vec,
                               result_images_countvect=result_images_countvect,
                               result_blend=top4_blend,
                               click_validation=True)


@app.route('/gallery_uploader', methods=['GET', 'POST'])
def gallery_uploader():
    """Look for images similar to the one clicked by user"""
    if request.method == 'POST':
        f = request.form['filename']
        query = request.form['query']
        image_directory = os.path.join(
            './app/static/images/room_scenes', os.path.basename(f))
        search_dir, engine, static_path, object_class, bound_boxes = get_engine(
            image_directory, app.config['ALLOWED_CLASSES'])
        if object_class == "all":
            object_image = image_directory
            object_image_path = os.path.join(
                'static/images/room_scenes', os.path.basename(image_directory))
            object_text = "room"
        else:
            object_image = crop_box_for_class(bound_boxes, image_directory, app.config[
                                              'BOUNDING_BOXES'], object_class)
            object_image_path = os.path.join(
                '/static/bounding_boxes', object_class + "_show_" + os.path.basename(image_directory))
            object_text = object_class
        predictions_path = os.path.join(
            app.config['YOLO_FOLDER'], 'predictions_' + os.path.basename(image_directory))
        similar_images = return_similar(object_image, search_dir, engine)
        result_images = [os.path.join(static_path, image)
                         for image in similar_images]
        # if object_class == "all":
        #     query_image = f
        # else:
        #     query_image = os.path.join(
        #         app.config['YOLO_FOLDER'], os.path.basename(predictions_path))

        if query == '':
            if object_class == 'pottedplant':
                images_countvect = search_engine.process_query('plant pot')
                images_w2vec = search_engine.process_query_w2vec('plant pot')
            else:
                images_countvect = search_engine.process_query(object_class)
                images_w2vec = search_engine.process_query_w2vec(object_class)
        else:
            images_countvect = search_engine.process_query(query)
            images_w2vec = search_engine.process_query_w2vec(query)

        # result_images_w2vec = [os.path.join('/static', image) for image in images_w2vec]
        # result_images_countvect = [os.path.join('/static', image) for image in images_countvect]

        result_images = [os.path.join(static_path, image)
                         for image in similar_images]
        top4_blend = result_images[:4] + \
            images_w2vec[:4] + images_countvect[:4]

        return render_template('pre_search_image.html',
                               title="Search for similar image",
                               query_image=os.path.join(
                                   '/static/yolo_detections/', os.path.basename(predictions_path)),
                               object_image=object_image_path,
                               object_class=object_text,
                               result_images=result_images,
                               text_query=query,
                               result_images_w2vec=images_w2vec,
                               result_images_countvect=images_countvect,
                               result_blend=top4_blend,
                               click_validation=True)


@app.route('/update_text', methods=['GET', 'POST'])
def update_text():
    if request.method == 'POST':
        query = request.form['query']
        query_image = request.form['query_image']
        object_image_path = request.form['object_image']
        object_class = request.form['object_class']
        result_images = [y.replace("'", "").replace("[", "").replace(
            "]", "") for y in request.form['result_images'].split(',')]
        print('result images', result_images)
        print(type(result_images))
        if query == '':
            if object_class == 'pottedplant':
                images_countvect = search_engine.process_query('plant pot')
                images_w2vec = search_engine.process_query_w2vec('plant pot')
            else:
                images_countvect = search_engine.process_query(object_class)
                images_w2vec = search_engine.process_query_w2vec(object_class)
        else:
            images_countvect = search_engine.process_query(query)
            images_w2vec = search_engine.process_query_w2vec(query)

        result_images_w2vec = [os.path.join(
            'app/', image) for image in images_w2vec]
        result_images_countvect = [os.path.join(
            'app/', image) for image in images_countvect]

        # top4_blend = result_images[:4] + result_images_w2vec[:4] + result_images_countvect[:4]
        blended_results = result_images[:4] + return_feature_blender_results(
            './app' + result_images[0],
            result_images_w2vec,
            result_images_countvect)
        print('blended results', blended_results)

        return render_template('search_image.html',
                               title="Search for similar image",
                               query_image=query_image,
                               object_image=object_image_path,
                               object_class=object_class,
                               result_images=result_images,
                               text_query=query,
                               result_images_w2vec=images_w2vec,
                               result_images_countvect=images_countvect,
                               result_blend=[y.replace("app/", "")
                                             for y in blended_results],
                               object_color=app.config['CLASS_COLORS'][object_class])


@app.route('/update_object', methods=['GET', 'POST'])
def update_object():
    if request.method == 'POST':
        click_is_valid = True
        image_x = request.form['form_x']
        image_y = request.form['form_y']
        image_dir = request.form['query_image']
        # text_query = request.form['text_query']
        image_width = request.form['width']
        object_image = request.form['object_image']
        previous_object = request.form['object_class']
        result_images = request.form['result_images']
        try:
            base_image_path = os.path.join(app.config['ROOM_DIR'], os.path.basename(
                image_dir).replace("predictions_", ""))
            bound_boxes = detect_objects_on_image(base_image_path)
            clicked_box = get_clicked_object(
                base_image_path, bound_boxes, image_x, image_y, image_width)
        except FileNotFoundError:
            base_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(
                image_dir).replace("predictions_", ""))
            bound_boxes = detect_objects_on_image(base_image_path)
            clicked_box = get_clicked_object(
                base_image_path, bound_boxes, image_x, image_y, image_width)
        if clicked_box != 0:
            object_class = clicked_box[0]
            cropped_image = crop_bounding_box_from_image(
                clicked_box, base_image_path, with_margin=False)
            cropped_image_path = os.path.join(
                app.config['BOUNDING_BOXES'], object_class + "_" + os.path.basename(image_dir))
            while os.path.isfile(cropped_image_path):
                cropped_image_path = os.path.join(
                    app.config['BOUNDING_BOXES'], "2" + os.path.basename(cropped_image_path))
            cropped_image.save(cropped_image_path)
            search_dir, engine, static_path = get_directories(object_class)
            similar_images = return_similar(
                cropped_image_path, search_dir, engine)
            result_images = [os.path.join(static_path, image)
                             for image in similar_images]
        else:
            # not a valid object clicked
            cropped_image_path = os.path.basename(object_image)
            object_class = previous_object
            click_is_valid = False
            result_images = [y.replace("'", "").replace("[", "").replace(
                "]", "") for y in request.form['result_images'].split(',')]
        images_countvect = search_engine.process_query(object_class)
        images_w2vec = search_engine.process_query_w2vec(object_class)
        # result_images_w2vec = [os.path.join('/static', image) for image in images_w2vec]
        # result_images_countvect = [os.path.join('/static', image) for image in images_countvect]
        top4_blend = result_images[:4] + \
            images_w2vec[:4] + images_countvect[:4]

        return render_template('search_image.html',
                               title="Search for similar image",
                               query_image=image_dir,
                               object_image=os.path.join(
                                   '/static/bounding_boxes', os.path.basename(cropped_image_path)),
                               object_class=object_class,
                               result_images=result_images,
                               # text_query=text_query,
                               result_blend=top4_blend,
                               click_validation=click_is_valid,
                               object_color=app.config['CLASS_COLORS'][object_class])


@app.route('/scene_gallery')
def scene_gallery():
    """Render template for gallery of scene images"""
    images = []
    for filename in os.listdir('./app/static/images/room_scenes'):
        if not filename.endswith('.jpg'):
            continue
        im = Image.open(os.path.join(
            './app/static/images/room_scenes', filename))
        w, h = im.size
        images.append({
            'width': int(w),
            'height': int(h),
            'src': os.path.join('./static/images/room_scenes', filename)
        })
    return render_template('scene_gallery.html',
                           images=images)
