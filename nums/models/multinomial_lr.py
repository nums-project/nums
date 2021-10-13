import numpy as np

from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication
from nums.core.application_manager import instance as _instance
from nums.core.array import utils as array_utils
from nums.core.array.random import NumsRandomState
from nums.core.linalg import inv

from nums.models.lbfgs import LBFGS
from nums.models.glms import Model


class MultinomialLogisticRegression(Model):
    def __init__(
        self,
        penalty="l2",
        C=1.0,
        tol=0.0001,
        max_iter=100,
        solver="newton-cg",
        lr=0.01,
        m=3,
        random_state=None,
        fit_intercept=True,
        normalize=False,
    ):

        if fit_intercept is False:
            raise NotImplementedError("fit_incercept=False currently not supported.")
        if normalize is True:
            raise NotImplementedError("normalize=True currently not supported.")

        self._app = _instance()
        if random_state is None:
            self.rs: NumsRandomState = self._app.random
        elif array_utils.is_int(random_state):
            self.rs: NumsRandomState = NumsRandomState(
                cm=self._app.cm, seed=random_state
            )
        elif isinstance(random_state, NumsRandomState):
            self.rs: NumsRandomState = random_state
        else:
            raise Exception(
                "Unexpected type for random_state %s" % str(type(random_state))
            )
        self._penalty = None if penalty == "none" else penalty
        if not (self._penalty is None or self._penalty == "l2"):
            raise NotImplementedError("%s penalty not supported" % self._penalty)
        self._lambda = 1.0 / C
        self._lambda_vec = None
        self._tol = tol
        self._max_iter = max_iter
        self._opt = solver
        self._lr = lr
        self._m = m
        self._beta = None
        self._beta0 = None

        self._num_class = None
        self.feature_dim = None
        self.feature_block_dim = None
        self.use_lbfgs_forward = False
        self._lambda_id = None

    def fit(self, X: BlockArray, y: BlockArray):
        # Note, it's critically important from a performance point-of-view
        # to maintain the original block shape of X below, along axis 1.
        # Otherwise, the concatenation operation will not construct the new X
        # by referencing X's existing blocks.
        # TODO: Option to do concat.
        # TODO: Provide support for batching.
        X = self._app.concatenate(
            [
                X,
                self._app.ones(
                    shape=(X.shape[0], 1),
                    block_shape=(X.block_shape[0], 1),
                    dtype=X.dtype,
                ),
            ],
            axis=1,
            axis_block_size=X.block_shape[1],
        )
        assert (
            len(X.shape) == 2 and len(y.shape) == 2
        ), "X must be a 2D matrix and Y must be one-hot"
        self._num_class = y.shape[1]

        self.feature_dim = X.shape[1]
        self.feature_block_dim = X.block_shape[1]

        beta: BlockArray = self._app.zeros(
            (X.shape[1], self._num_class),
            (X.block_shape[1], self._num_class),
            dtype=float,
        )
        tol: BlockArray = self._app.scalar(self._tol)
        max_iter: int = self._max_iter
        self.use_lbfgs_forward = False
        if self._penalty == "l2":
            self._lambda_vec = (
                self._app.ones(beta.shape, beta.block_shape, beta.dtype) * self._lambda
            )
        if self._opt == "gd" or self._opt == "sgd" or self._opt == "block_sgd":
            lr: BlockArray = self._app.scalar(self._lr)
            if self._opt == "gd":
                beta = gd(self, beta, X, y, tol, max_iter, lr)
            elif self._opt == "sgd":
                beta = sgd(self, beta, X, y, tol, max_iter, lr)
            else:
                beta = block_sgd(self, beta, X, y, tol, max_iter, lr)
        elif self._opt == "newton" or self._opt == "newton-cg":
            if self._penalty == "l2":
                self._lambda_id = (
                    self._app.eye(
                        (self.feature_dim, self.feature_dim),
                        block_shape=(self.feature_block_dim, self.feature_block_dim),
                    )
                    * self._lambda
                )
            beta = newton(self._app, self, beta, X, y, tol, max_iter)
        elif self._opt == "lbfgs":
            self.use_lbfgs_forward = True
            lbfgs_optimizer = LBFGS(
                model=self,
                m=self._m,
                max_iter=max_iter,
                thresh=self._tol,
                dtype=X.dtype,
            )
            beta = lbfgs_optimizer.execute(X, y, beta)
        else:
            raise Exception("Unsupported optimizer specified %s." % self._opt)
        self._beta0 = beta[-1]
        self._beta = beta[:-1]

    def lbfgs_forward(self, X, theta):
        if X.shape[1] < theta.shape[0]:
            assert X.shape[1] + 1 == theta.shape[0]
            eta = theta[-1] + X @ theta[:-1]
        else:
            eta = X @ theta
        eta = eta - self._app.max(eta, axis=1).expand_dims(-1)
        unnormalized_probs = self._app.exp(eta)
        mu = unnormalized_probs / self._app.sum(unnormalized_probs, axis=1).expand_dims(
            -1
        )
        # print('mu', mu.get()[0])
        return mu  # probabilities for each class

    def objective(
        self,
        X: BlockArray,
        y: BlockArray,
        beta: BlockArray = None,
        mu: BlockArray = None,
    ):
        assert beta is not None or self._beta is not None
        # neg log likelihood of correct class. y is an array of onehots
        return -self._app.sum(y * self._app.log(mu + 1e-10))

    def forward(self, X, beta=None):
        if self.use_lbfgs_forward:
            if beta:
                return self.lbfgs_forward(X, beta)
        if beta:
            return self.link_inv(X @ beta)
        return self.link_inv(self._beta0 + X @ self._beta)

    def link_inv(self, eta: BlockArray):
        def truncate(x, maximum):
            masked = (x - maximum) > 0
            return x * (1 - masked) + maximum * masked

        return self._app.one / (self._app.one + self._app.exp(truncate(-eta, 10)))

    def gradient(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        beta: BlockArray = None,
    ):
        if mu is None:
            mu = self.forward(X)
        if self._penalty is None:
            return X.T @ (mu - y)
        else:
            assert beta is not None
            return X.T @ (mu - y) + self._lambda_vec * beta

    def hessian(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        learning_ends_for_class=None,
    ):
        # pylint: disable=arguments-differ
        class_count = mu.shape[1]
        if mu is None:
            mu = self.forward(X)
        if learning_ends_for_class is None:
            learning_ends_for_class = [False for _ in range(class_count)]

        dim, block_dim = mu.shape[0], mu.block_shape[0]
        s = mu * (self._app.one - mu)
        if self._penalty is None:
            return [
                (
                    X.T
                    @ (
                        s[:, class_idx].reshape((dim, 1), block_shape=(block_dim, 1))
                        * X
                    )
                )
                if not learning_ends_for_class[class_idx]
                else None
                for class_idx in range(class_count)
            ]
        else:
            return [
                (
                    X.T
                    @ (
                        s[:, class_idx].reshape((dim, 1), block_shape=(block_dim, 1))
                        * X
                    )
                    + self._lambda_id
                )
                if not learning_ends_for_class[class_idx]
                else None
                for class_idx in range(class_count)
            ]

    def grad_norm_sq(self, X: BlockArray, y: BlockArray, beta=None):
        g = self.gradient(X, y, self.forward(X, beta), beta=beta)
        return self._app.sum(g * g)

    def predict(self, X: BlockArray):
        pred = self.forward(X).get()
        return np.argmax(pred, axis=-1)


def sgd(
    model: MultinomialLogisticRegression,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
    lr: BlockArray,
):
    # Classic SGD.
    app = _instance()
    for _ in range(max_iter):
        # Sample an entry uniformly at random.
        idx = model.rs.numpy().integers(X.shape[0])
        X_sample, y_sample = X[idx : idx + 1], y[idx : idx + 1]
        mu = model.forward(X_sample, beta)
        g = model.gradient(X_sample, y_sample, mu, beta=beta)
        beta += -lr * g
        if app.max(app.abs(g)) <= tol:
            # sklearn uses max instead of l2 norm.
            break
    return beta


def block_sgd(
    model: MultinomialLogisticRegression,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
    lr: BlockArray,
):
    # SGD with batches equal to block shape along first axis.
    app = _instance()
    for _ in range(max_iter):
        for (start, stop) in X.grid.grid_slices[0]:
            X_batch, y_batch = X[start:stop], y[start:stop]
            bsize = X_batch.shape[0]
            mu = model.forward(X_batch, beta)
            g = model.gradient(X_batch, y_batch, mu, beta=beta)
            beta += -lr * g / bsize
            if app.max(app.abs(g)) <= tol:
                return beta
    return beta


def gd(
    model: MultinomialLogisticRegression,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
    lr: BlockArray,
):
    app = _instance()

    for _ in range(max_iter):
        mu = model.forward(X, beta)
        g = model.gradient(X, y, mu, beta=beta)
        beta += -lr * g
        if app.max(app.abs(g)) <= tol:
            break
    return beta


def newton(
    app: ArrayApplication,
    model: MultinomialLogisticRegression,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
):
    num_classes = y.shape[1]
    learning_ends_for_class = [False for _ in range(num_classes)]

    opt_count = [0 for _ in range(num_classes)]
    for _ in range(max_iter):

        mu: BlockArray = model.forward(X, beta)
        g = model.gradient(X, y, mu, beta=beta)

        hessians = model.hessian(X, y, mu, learning_ends_for_class)

        class_count = g.shape[1]

        for class_idx in range(class_count):
            if learning_ends_for_class[class_idx]:
                continue
            opt_count[class_idx] += 1
            # These are PSD, but inv is faster than psd inv.

            h = hessians[class_idx]
            stable_h = h + app.eye(h.shape, h.block_shape) * 1e-6
            invert_stable_h = inv(app, stable_h)

            step = -invert_stable_h @ g[:, class_idx]
            beta[:, class_idx] += step  # - invert_stable_h @ g[:,class_idx]

            if app.max(app.abs(g[:, class_idx])) <= tol:
                learning_ends_for_class[class_idx] = True

        # learning ends if all class finishes
        if all(learning_ends_for_class):
            break
    return beta
