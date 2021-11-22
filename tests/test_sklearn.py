import pytest

from nums.core.array.blockarray import BlockArray
import nums.numpy as nps
from nums.core.array.application import ArrayApplication

# pylint: disable = import-outside-toplevel, unbalanced-tuple-unpacking


def sample_Xy(categorical=True, seed=1337, single_block=False):
    from nums.numpy.random import RandomState

    rs = RandomState(seed)

    size, feats = 10, 3
    block_shape = (size, feats) if single_block else (5, 2)
    X: BlockArray = rs.rand(size, feats).reshape(block_shape=block_shape)
    if categorical:
        y: BlockArray = rs.randint(2, size=size).reshape(block_shape=(block_shape[0],))
        y[0] = 1
        y[3] = 0
    else:
        y: BlockArray = rs.rand(
            size,
        ).reshape(block_shape=(block_shape[0],))
    return X, y


def classifier_pipelines(
    X, y, train_test_split, StandardScaler, RobustScaler, KNeighborsClassifier, SVC
):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1337)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1337)

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


def exec_parallel():
    from nums.sklearn import (
        train_test_split,
        StandardScaler,
        RobustScaler,
        KNeighborsClassifier,
        SVC,
        Ridge,
        ElasticNet,
    )

    X, y = sample_Xy(categorical=True)
    classifier_pipelines(
        X, y, train_test_split, StandardScaler, RobustScaler, KNeighborsClassifier, SVC
    )
    X, y = sample_Xy(categorical=False)
    regressor_pipelines(
        X, y, train_test_split, StandardScaler, RobustScaler, Ridge, ElasticNet
    )


def exec_serial():
    from sklearn import svm
    from sklearn import preprocessing
    from sklearn import neighbors
    from sklearn import model_selection
    from sklearn import linear_model

    X, y = sample_Xy(categorical=True)
    X, y = X.get(), y.get()
    classifier_pipelines(
        X,
        y,
        model_selection.train_test_split,
        preprocessing.StandardScaler,
        preprocessing.RobustScaler,
        neighbors.KNeighborsClassifier,
        svm.SVC,
    )

    X, y = sample_Xy(categorical=False)
    X, y = X.get(), y.get()
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
    exec_parallel()


def test_supervised(nps_app_inst: ArrayApplication):
    assert nps_app_inst is not None
    import numpy as np
    from sklearn import svm
    from sklearn import preprocessing
    from nums.sklearn import StandardScaler
    from nums.sklearn import SVC, SVR

    nps_X, nps_y = sample_Xy(single_block=True)
    X, y = nps_X.get(), nps_y.get()

    nps_pX = StandardScaler().fit_transform(nps_X)
    pX = preprocessing.StandardScaler().fit_transform(X)
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

    X, y = sample_Xy()
    assert not X.is_single_block(), not y.is_single_block()
    X_t, X_v, y_t, y_v = train_test_split(X, y, random_state=1337)
    assert (
        X_t.is_single_block()
        and X_v.is_single_block()
        and y_t.is_single_block()
        and y_v.is_single_block()
    )

    # Make sure rng works properly.
    X_t2, X_v2, y_t2, y_v2 = train_test_split(X, y, random_state=1337)
    assert nps.allclose(X_t, X_t2) and nps.allclose(X_v, X_v2)
    assert nps.allclose(y_t, y_t2) and nps.allclose(y_v, y_v2)

    X_t2, X_v2, y_t2, y_v2 = train_test_split(X, y, random_state=1338)
    assert not nps.allclose(X_t, X_t2) and not nps.allclose(X_v, X_v2)
    assert not nps.allclose(y_t, y_t2) and not nps.allclose(y_v, y_v2)

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
    assert nps_app_inst is not None
    from nums.sklearn import (
        DecisionTreeRegressor,
        LinearRegression,
    )

    regressors = [DecisionTreeRegressor, LinearRegression]
    X, y = sample_Xy(categorical=False, single_block=True)
    for Regressor in regressors:
        nps_model = Regressor()
        nps_model.fit(X, y)
        assert nps_model.predict(X).dtype is float


def test_typing(nps_app_inst):
    assert nps_app_inst is not None
    from nums import sklearn
    import numpy as np

    np_arr = np.arange(10)
    train, test = sklearn.train_test_split(np_arr, random_state=1337)
    assert isinstance(train, BlockArray) and isinstance(test, BlockArray)
    model = sklearn.Lasso()
    with pytest.raises(TypeError):
        model.fit(np_arr)

    X, y = sample_Xy(categorical=False, single_block=False)
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

    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    # test_parallel_sklearn(nps_app_inst)
    # test_supervised(nps_app_inst)
    test_train_test_split(nps_app_inst)
    # test_typing(nps_app_inst)
