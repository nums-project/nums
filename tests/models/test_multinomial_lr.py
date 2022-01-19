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


import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from nums.core.array.application import ArrayApplication
from nums.models.multinomial_lr import MultinomialLogisticRegression


@pytest.mark.parametrize("max_iter", [10, 2000])
def test_multinomial_logistic(nps_app_inst: ArrayApplication, max_iter):
    if max_iter > 100:
        pytest.skip("skipping long tests")
    real_X, real_y_indices = load_iris(return_X_y=True)
    num_samples, _, num_classes = (
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
        {"solver": "gd", "lr": 1e-6, "tol": 1e-8, "max_iter": max_iter},
        {"solver": "sgd", "lr": 1e-6, "tol": 1e-8, "max_iter": max_iter},
        {"solver": "block_sgd", "lr": 1e-6, "tol": 1e-8, "max_iter": max_iter},
        {"solver": "newton", "tol": 1e-8, "max_iter": max_iter},
        {"solver": "newton-cg", "tol": 1e-8, "max_iter": max_iter},
        {"solver": "lbfgs", "tol": 1e-8, "max_iter": max_iter},
        {"solver": "newton-cg", "tol": 1e-8, "max_iter": max_iter, "penalty": "none"},
        {"solver": "lbfgs", "tol": 1e-8, "max_iter": max_iter, "penalty": "none"},
    ]

    for kwargs in param_set:
        lr_model: MultinomialLogisticRegression = MultinomialLogisticRegression(
            **kwargs
        )
        lr_model.fit(X, y)
        y_pred = lr_model.predict(X)
        score = np.sum(y.get().argmax(axis=1) == y_pred) / num_samples
        # Sklearn multiclass lr only supports 'lbfgs', 'sag', 'saga' and 'newton-cg' solvers.
        if kwargs.get("solver") in ["lbfgs", "newton-cg"]:
            kwargs.update({"multi_class": "multinomial"})
            # pylint: disable=unexpected-keyword-arg
            clf = LogisticRegression(**kwargs).fit(real_X, real_y_indices)
            ref_score = clf.score(real_X, real_y_indices)
            # print("opt", kwargs["solver"])
            # print(score, ref_score)
            assert np.allclose(score, ref_score, atol=0.03)


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_multinomial_logistic(nps_app_inst, 1000)
