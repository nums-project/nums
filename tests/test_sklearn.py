import pytest

from nums.core.array.blockarray import BlockArray
import nums.numpy as nps
from nums.core.array.application import ArrayApplication

# pylint: disable = import-outside-toplevel, unbalanced-tuple-unpacking


def classifier_pipelines(
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


def regressor_pipelines(
    X, y, train_test_split, StandardScaler, RobustScaler, Ridge, ElasticNet
):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scores = []
    for preprocessor in [StandardScaler, RobustScaler]:
        p_inst = preprocessor()
        pX_train = p_inst.fit_transform(X_train)
        pX_test = p_inst.fit_transform(X_test)
        for solver in ["svd", "lsqr"]:
            m: Ridge = Ridge(solver=solver)
            m.fit(pX_train, y_train)
            scores.append(
                (
                    p_inst.__class__.__name__,
                    m.__class__.__name__,
                    m.score(pX_test, y_test),
                )
            )

        for l1_ratio in [0, 0.5, 1]:
            m: ElasticNet = ElasticNet(l1_ratio=l1_ratio)
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
        Ridge,
        ElasticNet,
    )

    X: BlockArray = nps.random.rand(size, features)
    y: BlockArray = nps.random.randint(2, size=size)
    classifier_pipelines(
        X, y, train_test_split, StandardScaler, RobustScaler, KNeighborsClassifier, SVC
    )
    y = nps.random.rand(size, 1)
    regressor_pipelines(
        X, y, train_test_split, StandardScaler, RobustScaler, Ridge, ElasticNet
    )


def exec_serial(size, features):
    import numpy as np
    from sklearn import svm
    from sklearn import preprocessing
    from sklearn import neighbors
    from sklearn import model_selection
    from sklearn import linear_model

    X = np.random.rand(size, features)
    y = np.random.randint(2, size=size)
    classifier_pipelines(
        X,
        y,
        model_selection.train_test_split,
        preprocessing.StandardScaler,
        preprocessing.RobustScaler,
        neighbors.KNeighborsClassifier,
        svm.SVC,
    )
    y = np.random.rand(size, 1)
    regressor_pipelines(
        X,
        y,
        model_selection.train_test_split,
        preprocessing.StandardScaler,
        preprocessing.RobustScaler,
        linear_model.Ridge,
        linear_model.ElasticNet,
    )


@pytest.mark.skip
def test_parallel_sklearn(nps_app_inst: ArrayApplication):
    assert nps_app_inst is not None

    size, feats = 10000, 10
    exec_parallel(size, feats)

    # t = time.time()
    # exec_parallel(size, feats)
    # print("parallel time", time.time() - t)

    # t = time.time()
    # exec_serial(size, feats)
    # print("serial time", time.time() - t)


def test_supervised(nps_app_inst: ArrayApplication):
    from nums.core.systems.systems import RaySystem, SerialSystem

    if not isinstance(nps_app_inst.cm.system, (RaySystem, SerialSystem)):
        return

    assert nps_app_inst is not None
    import numpy as np
    from sklearn import svm
    from sklearn import preprocessing
    from nums.sklearn import StandardScaler
    from nums.sklearn import SVC, SVR

    size, feats = 10, 10
    X = np.random.rand(size, feats)
    y = np.random.randint(2, size=size)
    pX = preprocessing.StandardScaler().fit_transform(X)

    nps_X = nps.array(X)
    nps_y = nps.array(y)
    nps_pX = StandardScaler().fit_transform(nps_X)
    assert np.allclose(pX, nps_pX.get())

    def test_model_pair(sklearn_cls, nps_cls):
        sklearn_model = sklearn_cls()
        sklearn_model.fit(pX, y)
        y_pred = sklearn_model.predict(pX)
        sklearn_score = sklearn_model.score(pX, y)

        nps_model = nps_cls()
        nps_model.fit(nps_pX, nps_y)
        nps_y_pred = nps_model.predict(nps_pX)
        nps_score = nps_model.score(nps_pX, nps_y)

        assert np.allclose(y_pred, nps_y_pred.get())
        assert np.allclose(sklearn_score, nps_score.get())

    test_model_pair(svm.SVC, SVC)
    test_model_pair(svm.SVR, SVR)


def test_train_test_split(nps_app_inst: ArrayApplication):
    assert nps_app_inst is not None
    import numpy as np
    from nums.sklearn import SVC, SVR, train_test_split

    size, feats = 100, 10
    X: BlockArray = nps.random.rand(size, feats).reshape(block_shape=(10, 10))
    y: BlockArray = nps.random.randint(2, size=size).reshape(block_shape=(10,))
    assert not X.is_single_block(), not y.is_single_block()
    X_t, X_v, y_t, y_v = train_test_split(X, y)
    assert (
        X_t.is_single_block()
        and X_v.is_single_block()
        and y_t.is_single_block()
        and y_v.is_single_block()
    )

    def test_model_pair(model_cls):
        model = model_cls()
        model.fit(X_t, y_t)
        y_v_pred = model.predict(X_v)
        assert y_v_pred.get().shape[0] == X_v.shape[0]
        model_score = model.score(X_v, y_v)
        assert isinstance(model_score.get(), np.ndarray)

    test_model_pair(SVC)
    test_model_pair(SVR)


def test_regressors(nps_app_inst: ArrayApplication):
    from nums.core.systems.systems import RaySystem, SerialSystem

    if not isinstance(nps_app_inst.cm.system, (RaySystem, SerialSystem)):
        return

    assert nps_app_inst is not None
    from nums.sklearn import (
        MLPRegressor,
        KNeighborsRegressor,
        SVR,
        GaussianProcessRegressor,
        DecisionTreeRegressor,
        RandomForestRegressor,
        AdaBoostRegressor,
        GradientBoostingRegressor,
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
    )

    regressors = [
        MLPRegressor,
        KNeighborsRegressor,
        SVR,
        GaussianProcessRegressor,
        DecisionTreeRegressor,
        RandomForestRegressor,
        AdaBoostRegressor,
        GradientBoostingRegressor,
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
    ]
    size, feats = 10, 3
    X: BlockArray = nps.random.rand(size, feats)
    y: BlockArray = nps.random.rand(size, 1)
    for Regressor in regressors:
        nps_model = Regressor()
        nps_model.fit(X, y)
        assert nps_model.predict(X).dtype is float


def test_typing(nps_app_inst):
    from nums.core.systems.systems import RaySystem, SerialSystem

    if not isinstance(nps_app_inst.cm.system, (RaySystem, SerialSystem)):
        return

    assert nps_app_inst is not None
    from nums import sklearn
    import numpy as np

    np_arr = np.arange(100)
    train, test = sklearn.train_test_split(np_arr)
    assert isinstance(train, BlockArray) and isinstance(test, BlockArray)
    model = sklearn.Lasso()
    with pytest.raises(TypeError):
        model.fit(np_arr)
    X = nps.array(np_arr).reshape((10, 10), block_shape=(5, 5)).astype(float)
    y = nps.arange(10).reshape(block_shape=(5,)).astype(float)
    with pytest.raises(ValueError):
        model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X)
    with pytest.raises(ValueError):
        model.fit_transform(X, y)
    with pytest.raises(ValueError):
        model.fit_transform(X, y)
    with pytest.raises(ValueError):
        model.score(X, y)


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "ray"
    nps_app_inst = application_manager.instance()
    # test_parallel_sklearn(nps_app_inst)
    test_supervised(nps_app_inst)
    test_train_test_split(nps_app_inst)
    # test_typing(nps_app_inst)
