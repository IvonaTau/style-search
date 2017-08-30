import gensim
import numpy as np
import os
import re
import string

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

class LabeledSentencesFromDictDoc2Vec(object):
    STOP_LIST = set('for a of the and to in view more product infromation \
                    an w very by has ikea get with as you it on thats have \
                    price reflects selected options'.split())
    MINOR_KEYS = ['name', 'color']
    MINOR_KEYS2 = ['type', 'desc']

    def __init__(self, dictname):
        self.dictname = dictname

    def __iter__(self):
        for key in list(self.dictname.keys()):

            for minor_key in self.MINOR_KEYS2:
                words = self.dictname[key][minor_key].lower().split()
                words_list = [
                    word for word in words if word not in self.STOP_LIST]

            for minor_key in self.MINOR_KEYS:
                words_list.append(self.dictname[key][minor_key].lower())

            words_list = [_f for _f in words_list if _f]

            yield gensim.models.doc2vec.LabeledSentence(words=words_list, tags=[key])
            

class SentencesWord2Vec(object):
    
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname)[1:]:
            sent = [line.strip() for line in open(os.path.join(self.dirname, fname))]
            yield sent

class CountVectModel(object):

    def __init__(self, products_dict):
        self.products_dict = products_dict

    def __preprocess_data(self):
        regex = " *[%s]+ *" % string.punctuation.replace("\\", "\\\\").replace("]", "\\]")
        descs_dict = {}
        stop_list = set('for a of the and to in view more product infromation \
                an w very by has ikea get with as information you it on thats have \
                price reflects selected options guarantee  brochure year read about terms'.split())

        for pr_id in list(self.products_dict.keys()):
            desc = self.products_dict[pr_id]['desc'] + ' ' + self.products_dict[pr_id]['type'] \
            + ' ' + self.products_dict[pr_id]['name'] + ' ' + self.products_dict[pr_id]['color']
            desc = desc.lower().split()
            desc = [word for word in desc if word not in stop_list]
            desc = ' '.join(desc)
            desc = re.sub(regex, " ", desc)
            descs_dict[pr_id] = desc

        return descs_dict

    def map_items_to_vectors(self):
        desc_dict_transformed = {}

        descs_dict = self.__preprocess_data()
        count_vect = CountVectorizer()
        descs_counts = count_vect.fit_transform(list(descs_dict.values()))
        svd = TruncatedSVD(n_components=25, n_iter=5)
        descs_counts_reduced = svd.fit_transform(descs_counts)
        for i, key in enumerate(descs_dict.keys()):
            desc_dict_transformed[key] = descs_counts_reduced[i]

        return {'transformed': desc_dict_transformed, 'counts': descs_counts, 'countvect': count_vect, 'svd': svd}

class MyModel(object):

    def __init__(self, labeled_products_filepath, model_name, model):
        self.labeled_products_filepath = labeled_products_filepath
        self.model_name = model_name
        self.model = model

    def _room_to_index(self, x):
        return {
            'bedroom': 1,
            'bathroom': 2,
            'home-office': 3,
            'baby-children-room': 4,
            'kitchen': 5,
            'living-room': 6,
            'dining': 7,
            'hallway': 8,
            'outdoor': 9,
            'laundry': 10,
        }[x]

    def _get_vectors(self):

        items = []
        labels = []
        vectors = []

        with open(self.labeled_products_filepath) as labeled_products:
            lines = labeled_products.readlines()

            for line in lines:
                item, label = line.split(';')
                items.append(item)
                slabel = label.strip()
                labels.append(self._room_to_index(slabel))

                if self.model_name == 'doc2vec':
                    vec = self.model.docvecs[item]
                    vectors.append(vec)

                elif self.model_name == 'word2vec' or self.model_name=='countvectorizer':
                    vec = self.model[item]
                    vectors.append(vec)

            return {'vectors': np.array(vectors),
                    'labels': np.array(labels), 'items': np.array(items)}