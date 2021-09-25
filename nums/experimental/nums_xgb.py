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


import logging
import time
import uuid
from threading import Thread
from typing import Dict

import numpy as np
import xgboost as xgb

import nums.numpy as nps
from nums.core.application_manager import instance as _instance
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray, Block
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid
from nums.core.systems import utils as systems_utils


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results."""
    # TODO (hme): Cleanup thread and tracker after training.
    host = systems_utils.get_private_ip()

    env = {"DMLC_NUM_WORKER": num_workers}
    rabit_tracker = xgb.RabitTracker(hostIP=host, nslave=num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.slave_envs())
    rabit_tracker.start(num_workers)

    # Wait until context completion
    thread = Thread(target=rabit_tracker.join)
    thread.daemon = True
    thread.start()

    return env


class RabitContext:
    """Context to connect a worker to a rabit tracker"""

    def __init__(self, actor_id, args):
        self.args = args
        self.args.append(("DMLC_TASK_ID=[xgboost.modin]:" + actor_id).encode())

    def __enter__(self):
        xgb.rabit.init(self.args)

    def __exit__(self, *args):
        xgb.rabit.finalize()


def xgb_train_remote(X, y, rabit_args, params, args, kwargs, *evals_flat):
    local_params = params
    dtrain = xgb.DMatrix(X, y)
    evals = []
    for i in range(0, len(evals_flat), 3):
        eval_X, eval_y, eval_method = evals_flat[i : i + 3]
        evals.append((xgb.DMatrix(eval_X, eval_y), eval_method))
    evals_result = dict()

    s = time.time()
    with RabitContext(uuid.uuid4().hex, rabit_args):
        bst = xgb.train(
            local_params,
            dtrain,
            *args,
            evals=evals,
            evals_result=evals_result,
            **kwargs
        )
        logging.getLogger(__name__).info("Local Train: {}".format(time.time() - s))
        return np.array({"bst": bst, "evals_result": evals_result}, dtype=dict)


def xgb_predict_remote(result, X):
    model = result.item()["bst"]
    y_pred = (model.predict(xgb.DMatrix(X)) > 0.5).astype(int)
    return y_pred


class NumsDMatrix(xgb.DMatrix):
    def __init__(self, X, y):
        super().__init__(None)
        self.X: BlockArray = X
        self.y: BlockArray = y

    def __iter__(self):
        yield self.X
        yield self.y


def train(params: Dict, data: NumsDMatrix, *args, evals=(), **kwargs):
    X: BlockArray = data.X
    y: BlockArray = data.y
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[0] and X.block_shape[0] == y.block_shape[0]
    assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)

    app: ArrayApplication = _instance()
    cm: ComputeManager = app.cm
    cm.register("xgb_train", xgb_train_remote, {})

    # Start tracker
    num_workers = X.grid.grid_shape[0]
    env = _start_rabit_tracker(num_workers)
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    evals_flat = []
    for eval_X, eval_y, eval_method in evals:
        if eval_X.shape != eval_X.block_shape:
            eval_X = eval_X.reshape(shape=eval_X.shape, block_shape=eval_X.shape)
        if eval_y.shape != eval_y.block_shape:
            eval_y = eval_y.reshape(shape=eval_y.shape, block_shape=eval_y.shape)
        eval_X_oid = eval_X.blocks.item().oid
        eval_y_oid = eval_y.blocks.item().oid
        evals_flat += [eval_X_oid, eval_y_oid, eval_method]

    X: BlockArray = X.reshape(block_shape=(X.block_shape[0], X.shape[1]))
    result: BlockArray = BlockArray(
        ArrayGrid(shape=(X.grid.grid_shape[0],), block_shape=(1,), dtype="dict"), cm
    )
    for grid_entry in X.grid.get_entry_iterator():
        X_block: Block = X.blocks[grid_entry]
        i = grid_entry[0]
        if len(y.shape) == 1:
            y_block: Block = y.blocks[i]
        else:
            y_block: Block = y.blocks[i, 0]
        syskwargs = {"grid_entry": grid_entry, "grid_shape": X.grid.grid_shape}
        result.blocks[i].oid = cm.call(
            "xgb_train",
            X_block.oid,
            y_block.oid,
            rabit_args,
            params,
            args,
            kwargs,
            *evals_flat,
            syskwargs=syskwargs
        )
    return result


class XGBClassifier(object):
    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        tree_method="approx",
        objective="binary:logistic",
        booster="gbtree",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.tree_method = tree_method
        self.objective = objective
        self.booster = booster
        self.model: BlockArray = None

    def fit(self, X: BlockArray, y: BlockArray):
        dtrain = NumsDMatrix(X, y)
        self.model = train(
            {
                "max_depth": self.max_depth,
                "eta": self.learning_rate,
                "tree_method": self.tree_method,
                "booster": self.booster,
                "objective": self.objective,
            },
            dtrain,
            self.n_estimators,
        )
        return self

    def predict(self, X: BlockArray):
        app: ArrayApplication = _instance()
        cm: ComputeManager = app.cm
        cm.register("xgb_predict", xgb_predict_remote, {})
        model_block: Block = self.model.blocks[0]
        result: BlockArray = BlockArray(
            ArrayGrid(
                shape=(X.shape[0],),
                block_shape=(X.block_shape[0],),
                dtype=int.__name__,
            ),
            cm,
        )
        for grid_entry in X.grid.get_entry_iterator():
            i = grid_entry[0]
            X_block: Block = X.blocks[grid_entry]
            r_block: Block = result.blocks[i]
            syskwargs = {"grid_entry": grid_entry, "grid_shape": X.grid.grid_shape}
            r_block.oid = cm.call(
                "xgb_predict", model_block.oid, X_block.oid, syskwargs=syskwargs
            )
        return result


if __name__ == "__main__":
    from nums.core import settings
    import nums

    filename = settings.pj(
        settings.project_root, "tests", "core", "storage", "test.csv"
    )
    X: BlockArray = nums.read_csv(filename, has_header=True)
    y: BlockArray = nps.random.random_sample(X.shape[0])
    model = XGBClassifier()
    model.fit(X, y)
    print(model.predict(X).get())
