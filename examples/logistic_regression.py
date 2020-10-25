from nums import numpy as nps
from nums.core.models import LogisticRegression


# TODO (hme): Rewrite GLM API to match sklearn.


# Make dataset.
rs = nps.random.RandomState(1337)
X1 = rs.randn(500, 1) + 5.0
y1 = nps.zeros(shape=(500,), dtype=bool)
X2 = rs.randn(500, 1) + 10.0
y2 = nps.ones(shape=(500,), dtype=bool)
X = nps.concatenate([X1, X2], axis=0)
y = nps.concatenate([y1, y2], axis=0)


# Train Logistic Regression Model.
model = LogisticRegression(nps.app, opt="newton", opt_params={"tol": 1e-8, "max_iter": 1})
model.fit(X, y)
y_pred = model.predict(X) > 0.5
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
