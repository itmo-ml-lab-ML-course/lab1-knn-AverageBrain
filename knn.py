import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import linalg
from scipy import stats
import pandas as pd
import random
from typing import Tuple, Any, List


class KNNClassifier:
    weights: np.ndarray = None
    kernels = {
        'uniform': lambda x: stats.uniform.pdf(x, loc=-1, scale=2),
        'gaussian': lambda x: 1 / math.sqrt(2 * math.pi) * math.exp(- (x ** 2 / 2)),
        'triangular': lambda x: max(0, 1 - abs(x)),
        'epanechnikov': lambda x: max(0, 3 / 4 * (1 - x ** 2))
    }
    metrics = ['manhattan', 'euclidean', 'cosine']
    xs: pd.DataFrame = None
    ys: np.ndarray = None
    n_classes: int = 0

    def __init__(
            self,
            k: int = 1,
            window_type: str = 'non_fixed',
            window_param: float | None = None,
            metric: str = 'euclidean',
            kernel: str = 'gaussian',
            leaf_size: int = 30
    ):
        self.k = k
        self.window = None if window_type != 'fixed' else window_param
        self.leaf_size = leaf_size
        self.metric = metric
        self.kernel = self.kernels[kernel]

    def fit(self, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray = None):
        self.xs = x
        self.ys = y
        self.n_classes = len(np.unique(y))
        self.weights = weights if weights is not None else np.array([1.0] * len(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        cnt = self.k + 1 if self.window is None else self.k
        distances, ids = NearestNeighbors(n_neighbors=cnt, metric=self.metric)\
            .fit(self.xs)\
            .kneighbors(x, n_neighbors=cnt)
        classes = [self.ys[i] for i in ids]
        weights = [[self.weights[j] for j in i] for i in ids]

        answers = np.array([0] * len(x))
        for i in range(len(x)):
            d, c, w = distances[i], classes[i], weights[i]
            scores = [0] * self.n_classes
            for j in range(len(d) - 1):
                kernel_arg = d[j] / (self.window if self.window else d[-1])
                scores[c[j]] += self.kernel(kernel_arg) * w[j]
            answers[i] = scores.index(max(scores))

        return answers
