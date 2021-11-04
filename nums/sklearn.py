import warnings

from nums.core.array.blockarray import BlockArray
from nums.core.application_manager import (
    instance,
    call_on_create,
    ArrayApplication,
)


# pylint: disable = import-outside-toplevel


def _register_train_test_split(app: ArrayApplication):
    from sklearn import model_selection

    app.cm.register("train_test_split", model_selection.train_test_split, {})


call_on_create(_register_train_test_split)


def _check_array(array, strict=False):
    if not isinstance(array, BlockArray):
        if strict:
            raise TypeError("Input array is not a BlockArray.")
        # These arrays should be a single block.
        array = instance().array(array, block_shape=array.shape)
    if not array.is_single_block():
        if strict:
            raise ValueError("Input array is not a single block.")
        array_size_gb = array.nbytes / 10 ** 9
        if array_size_gb > 100.0:
            raise MemoryError(
                "Operating on an "
                "array of size %sGB is not supported." % array_size_gb
            )
        elif array_size_gb > 10.0:
            # This is a large array of size 10GB.
            warnings.warn(
                "Attempting to convert an array "
                "of size %sGB to a single block." % array_size_gb
            )
        array = array.to_single_block()
    return array


def train_test_split(*arrays, **options):
    # TODO: Add proper seed support.
    updated_arrays = []
    for array in arrays:
        updated_arrays.append(_check_array(array))
    kwargs = options.copy()
    kwargs["syskwargs"] = {
        "options": {"num_returns": 2 * len(updated_arrays)},
        "grid_entry": (0,),
        "grid_shape": (1,),
    }
    array_oids = [array.flattened_oids()[0] for array in updated_arrays]
    result_oids = instance().cm.call("train_test_split", *array_oids, **kwargs)
    # Optimize by computing this directly.
    shape_dtype_oids = [
        instance().cm.shape_dtype(
            r_oid, syskwargs={"grid_entry": (0,), "grid_shape": (1,)}
        )
        for r_oid in result_oids
    ]
    shape_dtypes = instance().cm.get(shape_dtype_oids)
    results = []
    for i, r_oid in enumerate(result_oids):
        shape, dtype = shape_dtypes[i]
        results.append(
            BlockArray.from_oid(r_oid, shape=shape, dtype=dtype, cm=instance().cm)
        )
    return results


def build_sklearn_actor(cls: type):
    from sklearn.base import ClassifierMixin, RegressorMixin

    name = cls.__name__
    predict_dtype = None
    if issubclass(cls, ClassifierMixin):
        predict_dtype = int
    elif issubclass(cls, RegressorMixin):
        predict_dtype = float

    # NOTE:
    # A possibly cleaner way of building actor classes is to check all the superclasses, then
    # procedurally add the methods inherited from those classes. Many superclasses can be found in
    # sklearn/base.py, and some in other subpackages. For example,
    # TransformerMixin: add fit_transform(), transform() (fit_transform calls transform)
    # ClusterMixin: add fit_predict()
    # LinearClassifierMixin: add decision_function()
    # BaseEstimator: add get_params(), set_params()

    class ModelActor(object):
        def __init__(self, *args, **kwargs):
            self.instance = cls(*args, **kwargs)

        def fit(self, X, y):
            self.instance = self.instance.fit(X, y)

        def fit_transform(self, X, y=None):
            return self.instance.fit_transform(X, y)

        def predict(self, X):
            return self.instance.predict(X)

        def score(self, X, y, sample_weight=None):
            return self.instance.score(X, y, sample_weight)

    class NumsModel(object):
        def __init__(self, *args, **kwargs):
            device_id = None
            if self.__class__ in _place_on_node_0:
                device_id = instance().cm.devices()[0]
            self.actor = instance().cm.make_actor(
                name, *args, device_id=device_id, **kwargs
            )

        # TODO: (all functions) test inputs are single block, if not warn about performance
        def fit(self, X: BlockArray, y: BlockArray):
            _check_array(X, True)
            _check_array(y, True)
            instance().cm.call_actor_method(
                self.actor, "fit", X.flattened_oids()[0], y.flattened_oids()[0]
            )

        def fit_transform(self, X: BlockArray, y: BlockArray = None):
            _check_array(X, True)
            if y is not None:
                _check_array(y, True)
                y = y.flattened_oids()[0]
            r_oid = instance().cm.call_actor_method(
                self.actor, "fit_transform", X.flattened_oids()[0], y
            )
            return BlockArray.from_oid(
                r_oid, shape=X.shape, dtype=float, cm=instance().cm
            )

        def predict(self, X: BlockArray):
            _check_array(X, True)
            r_oid = instance().cm.call_actor_method(
                self.actor, "predict", X.flattened_oids()[0]
            )
            return BlockArray.from_oid(
                r_oid, shape=(X.shape[0],), dtype=predict_dtype, cm=instance().cm
            )

        def score(self, X: BlockArray, y: BlockArray, sample_weight: BlockArray = None):
            _check_array(X, True)
            _check_array(y, True)
            if sample_weight is not None:
                _check_array(sample_weight, True)
                sample_weight = sample_weight.flattened_oids()[0]
            r_oid = instance().cm.call_actor_method(
                self.actor,
                "score",
                X.flattened_oids()[0],
                y.flattened_oids()[0],
                sample_weight,
            )
            return BlockArray.from_oid(r_oid, shape=(), dtype=float, cm=instance().cm)

    NumsModel.__name__ = "Nums" + cls.__name__
    ModelActor.__name__ = cls.__name__ + "Actor"
    call_on_create(lambda app: app.cm.register_actor(name, ModelActor))
    return NumsModel


def build_supervised_actors():
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.gaussian_process import (
        GaussianProcessClassifier,
        GaussianProcessRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestRegressor,
        AdaBoostRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import (
        LogisticRegression,
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
    )

    skl_models = [
        MLPClassifier,
        MLPRegressor,
        KNeighborsClassifier,
        KNeighborsRegressor,
        SVC,
        SVR,
        GaussianProcessClassifier,
        GaussianProcessRegressor,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        AdaBoostClassifier,
        AdaBoostRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        GaussianNB,
        BernoulliNB,
        QuadraticDiscriminantAnalysis,
        LogisticRegression,
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
    ]
    return (build_sklearn_actor(skl_model) for skl_model in skl_models)


(
    MLPClassifier,
    MLPRegressor,
    KNeighborsClassifier,
    KNeighborsRegressor,
    SVC,
    SVR,
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    GaussianNB,
    BernoulliNB,
    QuadraticDiscriminantAnalysis,
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
) = build_supervised_actors()


def build_preprocessors():
    from sklearn.preprocessing import StandardScaler, RobustScaler

    sklearn_clses = [StandardScaler, RobustScaler]
    return (build_sklearn_actor(sklearn_cls) for sklearn_cls in sklearn_clses)


(StandardScaler, RobustScaler) = build_preprocessors()
_place_on_node_0 = (StandardScaler, RobustScaler)


def expose_sklearn_objects():
    from sklearn.gaussian_process.kernels import RBF

    return (RBF,)


(RBF,) = expose_sklearn_objects()


if __name__ == "__main__":
    pass
