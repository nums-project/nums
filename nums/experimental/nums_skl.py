import time

import numpy as np
from sklearn import svm
import ray

from nums.core.array.blockarray import BlockArray
from nums.core.application_manager import instance
import nums.numpy as nps


@ray.remote
class SVCActor(object):

    def __init__(self, C, kernel, degree,
                 gamma, coef0, shrinking, probability,
                 tol, cache_size, class_weight, verbose,
                 max_iter, decision_function_shape,
                 break_ties, random_state):
        self.instance = svm.SVC(C, kernel, degree,
                                gamma, coef0, shrinking, probability,
                                tol, cache_size, class_weight, verbose,
                                max_iter, decision_function_shape,
                                break_ties, random_state)

    def fit(self, X, y, sample_weight=None):
        self.instance = self.instance.fit(X, y, sample_weight)

    def predict(self, X):
        return self.instance.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.instance.score(X, y, sample_weight)


class RemoteSVC(object):
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=- 1,
                 decision_function_shape='ovr', break_ties=False,
                 random_state=None):
        self.actor = SVCActor.remote(C, kernel, degree,
                                     gamma, coef0, shrinking, probability,
                                     tol, cache_size, class_weight, verbose,
                                     max_iter, decision_function_shape,
                                     break_ties, random_state)

    def fit(self, X: BlockArray, y: BlockArray, sample_weight: BlockArray = None):
        if sample_weight is not None:
            sample_weight = sample_weight.flattened_oids()[0]
        self.actor.fit.remote(X.flattened_oids()[0], y.flattened_oids()[0], sample_weight)

    def predict(self, X: BlockArray):
        y = self.actor.predict.remote(X.flattened_oids()[0])
        return BlockArray.from_oid(y, shape=(X.shape[0],), dtype=int, cm=instance().cm)

    def score(self, X: BlockArray, y: BlockArray, sample_weight: BlockArray = None):
        if sample_weight is not None:
            sample_weight = sample_weight.flattened_oids()[0]
        r = self.actor.score.remote(X.flattened_oids()[0], y.flattened_oids()[0], sample_weight)
        return BlockArray.from_oid(r, shape=(), dtype=float, cm=instance().cm)


def exec_parallel(size, features):
    X: BlockArray = nps.random.rand(size, features)
    y: BlockArray = nps.random.randint(2, size=size)

    models = []
    preds = []
    scores = []
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        m: RemoteSVC = RemoteSVC(kernel=kernel)
        m.fit(X, y)
        preds.append(m.predict(X))
        scores.append(m.score(X, y))
        models.append(m)
    for score in scores:
        print(score.get())


def exec_serial(size, features):
    X = np.random.rand(size, features)
    y = np.random.randint(2, size=size)

    models = []
    preds = []
    scores = []
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        m: svm.SVC = svm.SVC(kernel=kernel)
        m.fit(X, y)
        preds.append(m.predict(X))
        scores.append(m.score(X, y))
        models.append(m)
    for score in scores:
        print(score)


def example():
    size, feats = 10000, 10
    exec_parallel(size, feats)

    t = time.time()
    exec_parallel(size, feats)
    print("parallel time", time.time() - t)

    t = time.time()
    exec_serial(size, feats)
    print("serial time", time.time() - t)


if __name__ == "__main__":
    example()
