import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn.manifold import TSNE


class TSNEEmbeddingPlot(object):
    ROOM_LABELS = ['bedroom',
                   'bathroom',
                   'home-office',
                   'baby-children-room',
                   'kitchen',
                   'living-room',
                   'dining',
                   'hallway',
                   'outdoor',
                   'laundry']

    def __init__(self, vectors_array, labels_array, products_ids):
        self.vectors_array = vectors_array
        self.X_tsne = TSNE(perplexity=80, method='exact').fit_transform(self.vectors_array)
        self.labels_array = labels_array
        self.products_ids = products_ids

    def __create_list_of_scatters(self):
        list_of_plotly_scatters = []

        labeled_points = zip(X_tsne, labels_array)
        for l in labels_array:
            points = [point for (point, label) in labeled_points if label == l]
            current_scatter = go.Scattergl(
                x, y = zip(*points),
                name = self.ROOM_LABELS[l-1],
                mode = 'markers',
                )
            list_of_plotly_scatters.append(current_scatter)

        return list_of_plotly_scatters

    def generate_plot(self):
        py.sign_in('aleksandramozejko', 'b97RGZcGRgaAB65HuiHE')
        layout = dict(title='Styled Scatter',
                      yaxis=dict(zeroline=False),
                      xaxis=dict(zeroline=False))
        list_of_plotly_scatters = self.__create_list_of_scatters()
        fig = dict(data=list_of_plotly_scatters, layout=layout)
        url = py.plot(fig, filename='TSNE-viz-after-code-refactoring')
        return url