from nums.core.array.blockarray import BlockArray
from nums.core.application_manager import instance, RaySystem
import nums.numpy as nps

from sklearn.gaussian_process.kernels import RBF


def _register_train_test_split():
    from sklearn import model_selection

    instance().cm.register("train_test_split", model_selection.train_test_split, {})


_register_train_test_split()


def train_test_split(*arrays, **options):
    # TODO: Add proper seed support.
    for array in arrays:
        assert isinstance(array, BlockArray)
        assert array.is_single_block()
    kwargs = options.copy()
    kwargs["syskwargs"] = {
        "options": {"num_returns": 2 * len(arrays)},
        "grid_entry": (0,),
        "grid_shape": (1,),
    }
    array_oids = [array.flattened_oids()[0] for array in arrays]
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


def register_actor(name: str, cls: type):
    assert isinstance(instance().cm.system, RaySystem)
    sys: RaySystem = instance().cm.system
    sys.register_actor(name, cls)


def make_actor(name: str, *args, device_id, **kwargs):
    assert isinstance(instance().cm.system, RaySystem)
    sys: RaySystem = instance().cm.system
    return sys.make_actor(name, *args, device_id=device_id, **kwargs)


def build_sklearn_actor(cls: type):
    name = cls.__name__

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
            self.actor = make_actor(name, *args, device_id=device_id, **kwargs)

        def fit(self, X: BlockArray, y: BlockArray):
            self.actor.fit.remote(X.flattened_oids()[0], y.flattened_oids()[0])

        def fit_transform(self, X: BlockArray, y: BlockArray = None):
            if y is not None:
                y = y.flattened_oids()[0]
            r_oid = self.actor.fit_transform.remote(X.flattened_oids()[0], y)
            return BlockArray.from_oid(
                r_oid, shape=X.shape, dtype=float, cm=instance().cm
            )

        # TODO: Note the returned dtype => This is the right interface for CLASSIFIERS only.
        def predict(self, X: BlockArray):
            r_oid = self.actor.predict.remote(X.flattened_oids()[0])
            return BlockArray.from_oid(
                r_oid, shape=(X.shape[0],), dtype=int, cm=instance().cm
            )

        def score(self, X: BlockArray, y: BlockArray, sample_weight: BlockArray = None):
            if sample_weight is not None:
                sample_weight = sample_weight.flattened_oids()[0]
            r_oid = self.actor.score.remote(
                X.flattened_oids()[0], y.flattened_oids()[0], sample_weight
            )
            return BlockArray.from_oid(r_oid, shape=(), dtype=float, cm=instance().cm)

    ModelActor.__name__ = cls.__name__ + "Actor"
    NumsModel.__name__ = "Nums" + cls.__name__
    register_actor(name, ModelActor)
    return NumsModel


def build_classifier_actors():
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    assert isinstance(instance().cm.system, RaySystem)
    skl_models = [
        MLPClassifier,
        KNeighborsClassifier,
        SVC,
        GaussianProcessClassifier,
        DecisionTreeClassifier,
        RandomForestClassifier,
        AdaBoostClassifier,
        GaussianNB,
        QuadraticDiscriminantAnalysis,
    ]
    return (build_sklearn_actor(skl_model) for skl_model in skl_models)


(
    MLPClassifier,
    KNeighborsClassifier,
    SVC,
    GaussianProcessClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GaussianNB,
    QuadraticDiscriminantAnalysis,
) = build_classifier_actors()


def build_preprocessors():
    from sklearn.preprocessing import StandardScaler, RobustScaler

    sklearn_clses = [StandardScaler, RobustScaler]
    return (build_sklearn_actor(sklearn_cls) for sklearn_cls in sklearn_clses)


(StandardScaler, RobustScaler) = build_preprocessors()
_place_on_node_0 = (StandardScaler, RobustScaler)


if __name__ == "__main__":
    pass
