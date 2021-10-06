# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time

import numpy as np
import pytest

from sklearn.datasets import load_iris

from nums.core.array.application import ArrayApplication
from nums.core.storage.storage import BimodalGaussian
from nums.models.multinomial_lr import MultinomialLogisticRegression

# pylint: disable = protected-access, import-outside-toplevel, import-error


def test_multinomial_logistic(nps_app_inst: ArrayApplication):
    data = load_iris()
    real_X = data["data"]
    real_y_indices = data["target"]
    num_samples, num_features, num_classes = (
        real_X.shape[0],
        real_X.shape[1],
        real_y_indices.max() + 1,
    )
    real_y = np.zeros((num_samples, num_classes))
    real_y[np.arange(num_samples), real_y_indices] = 1  # make it a onehot
    X = nps_app_inst.array(real_X, block_shape=(100, 3))
    y = nps_app_inst.array(
        real_y, block_shape=(100, 3)
    )  # TODO block shape? iris is 3 classes, and we seem to crash when using less than 3 here.
    param_set = [
        # {"solver": "gd", "lr": 1e-6, "tol": 1e-8, "max_iter": 10},
        # {"solver": "sgd", "lr": 1e-6, "tol": 1e-8, "max_iter": 10},
        # {"solver": "block_sgd", "lr": 1e-6, "tol": 1e-8, "max_iter": 10},
        # {"solver": "newton", "tol": 1e-8, "max_iter": 10},
        # TODO: This is not working.
        {"solver": "lbfgs", "tol": 1e-8, "max_iter": 10, "m": 3}
    ]
    for kwargs in param_set:
        runtime = time.time()
        lr_model: MultinomialLogisticRegression = MultinomialLogisticRegression(
            **kwargs
        )
        lr_model.fit(X, y)
        runtime = time.time() - runtime
        y_pred = lr_model.predict(
            X
        )  # .get() TODO we should return a nums object not np
        # y_pred_proba = lr_model.predict_proba(X).get() # TODO this isn't implemented atm. does it make sense to implement?
        # np.allclose(np.ones(shape=(y.shape[0],)), y_pred_proba[:, 0] + y_pred_proba[:, 1]) # TODO not sure if we need this line
        print("opt", kwargs["solver"])
        print("runtime", runtime)
        # print("norm", lr_model.grad_norm_sq(X, y).get()) # TODO does this matter?
        # print("objective", lr_model.objective(X, y).get()) # TODO we don't have this function implemented
        print("accuracy", np.sum(y.get().argmax(axis=1) == y_pred) / num_samples)


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager

    nps_app_inst = application_manager.instance()
    test_multinomial_logistic(nps_app_inst)
