import numpy as np
from sklearn.decomposition import TruncatedSVD


class Blender(object):

	def __init__(self):
		pass

	def transform_img_features(features_norm):
    	svd = TruncatedSVD(n_components=25, n_iter=5)
    	features_reduced = svd.fit_transform(features_norm)
    	return features_reduced