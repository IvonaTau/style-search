import numpy as np
import operator
import re
import string
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import time
import logging


class SearchEngine(object):

    def __init__(self, products_dict,vectorizer, word2vec=None):
        self.products_dict = products_dict
        self.vectorizer = vectorizer
        self.word2vec = word2vec
        countvect_model = vectorizer.map_items_to_vectors()
        self.transformed = countvect_model['transformed']
        self.countvect = countvect_model['countvect']
        self.counts = countvect_model['counts']
        self.svd = countvect_model['svd']

    def __preprocess(self, text):
        regex = " *[%s]+ *" % string.punctuation.replace("\\", "\\\\").replace("]", "\\]")
        descs_dict = {}
        stop_list = set('for a of the and to in view more product infromation \
                an w very by has ikea get with as information you it on thats have \
                price reflects selected options guarantee  brochure year read about terms'.split())

        text = text.lower().split()
        text = [word for word in text if word not in stop_list]
        text = ' '.join(text)
        text = re.sub(regex, " ", text)
        return text
    
    def __get_key(self, closest):
        for key, value in self.transformed.items():
            if np.array_equal(np.array(value), closest):
                return key

    def __find_closest(self, text):
        cosine_sim = 1.0
        sel_elem = np.array([0])
        new_text = self.countvect.transform([self.__preprocess(text)])
        new_reduced = self.svd.transform(new_text)[0]

        for elem in list(self.transformed.values()):
            new_cosine = cosine(new_reduced, np.array(elem))
            if new_cosine < cosine_sim:
                cosine_sim = new_cosine
                sel_elem = elem
                item_id = self.__get_key(sel_elem)
        
        return (item_id, cosine_sim)

    def __find_n_closest(self, text, n):
        new_text = self.countvect.transform([self.__preprocess(text)])
        new_reduced = self.svd.transform(new_text)[0]
        cosine_sim = 1.0    
        all_elems = []
        all_sims = []
        items_ids = []
    
        for elem in list(self.transformed.values()):
            new_cosine = cosine(new_reduced, np.array(elem))
            all_sims.append(new_cosine)
            all_elems.append(elem)
        indices = np.argpartition(np.array(all_sims), n)
        sel_sims = np.array(all_sims)[indices[:n]]
        sel_elems = np.array(all_elems)[indices[:n]]

        for vec in sel_elems:
            item_id = self.__get_key(vec)
            items_ids.append(item_id)
            items_dict = dict(zip(items_ids, sel_sims))
            items_sorted = sorted(items_dict.items(), key=operator.itemgetter(1), reverse=True)
            items, sims = zip(*items_sorted)
        
        return (items, sims)

    def __get_images_paths(self, items_ids):
        try:
            paths = ["static/" + self.products_dict[item]['img'] for item in items_ids if self.products_dict[item]['img']]
        except KeyError:
            items_ids = [k for k in list(self.products_dict.keys())[:10]]
            paths = ["static/" + self.products_dict[item]['img'] for item in items_ids if self.products_dict[item]['img']]
        return paths

    def process_query(self, text):
        items_ids = self.__find_n_closest(text, 10)[0]
        return self.__get_images_paths(items_ids)

    def process_query_w2vec(self, text):
        try:
            item_id = self.__find_closest(text)[0]
            most_similar = self.word2vec.most_similar(positive = [item_id])
            most_similar = list(list(zip(*most_similar))[0])[:10]
        except UnboundLocalError:
            most_similar = [k for k in list(self.products_dict.keys())[:10]]
        return self.__get_images_paths(most_similar)