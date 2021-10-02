import time

from nums.core.array.blockarray import BlockArray
import nums.numpy as nps
from nums.core.array.application import ArrayApplication


def exec(
    X, y, train_test_split, StandardScaler, RobustScaler, KNeighborsClassifier, SVC
):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scores = []
    for preprocessor in [StandardScaler, RobustScaler]:
        p_inst = preprocessor()
        pX_train = p_inst.fit_transform(X_train)
        pX_test = p_inst.fit_transform(X_test)
        for kernel in ["linear", "rbf", "sigmoid"]:
            m: SVC = SVC(kernel=kernel)
            m.fit(pX_train, y_train)
            scores.append(
                (
                    p_inst.__class__.__name__,
                    m.__class__.__name__,
                    m.score(pX_test, y_test),
                )
            )

        for n_neighbors in [3, 5]:
            m: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            m.fit(pX_train, y_train)
            scores.append(
                (
                    p_inst.__class__.__name__,
                    m.__class__.__name__,
                    m.score(pX_test, y_test),
                )
            )

    for pname, mname, score in scores:
        if isinstance(score, BlockArray):
            score = score.get()
        print(pname, mname, score)


def exec_parallel(size, features):
    from nums.sklearn import (
        train_test_split,
        StandardScaler,
        RobustScaler,
        KNeighborsClassifier,
        SVC,
    )

    X: BlockArray = nps.random.rand(size, features)
    y: BlockArray = nps.random.randint(2, size=size)
    return exec(
        X, y, train_test_split, StandardScaler, RobustScaler, KNeighborsClassifier, SVC
    )


def exec_serial(size, features):
    import numpy as np
    from sklearn import svm
    from sklearn import preprocessing
    from sklearn import neighbors
    from sklearn import model_selection

    X = np.random.rand(size, features)
    y = np.random.randint(2, size=size)
    return exec(
        X,
        y,
        model_selection.train_test_split,
        preprocessing.StandardScaler,
        preprocessing.RobustScaler,
        neighbors.KNeighborsClassifier,
        svm.SVC,
    )


def test_parallel_sklearn(nps_app_inst: ArrayApplication):
    from nums.core.systems.systems import RaySystem

    if not isinstance(nps_app_inst.cm.system, RaySystem):
        return
    size, feats = 10000, 10
    exec_parallel(size, feats)

    # t = time.time()
    # exec_parallel(size, feats)
    # print("parallel time", time.time() - t)
    #
    # t = time.time()
    # exec_serial(size, feats)
    # print("serial time", time.time() - t)


if __name__ == "__main__":
    pass
    # test_parallel_sklearn()
