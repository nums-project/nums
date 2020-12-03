from nums import numpy as nps
from nums.models.glms import LogisticRegression


# Make dataset.
X1 = nps.random.randn(500, 1) + 5.0
y1 = nps.zeros(shape=(500,), dtype=bool)
X2 = nps.random.randn(500, 1) + 10.0
y2 = nps.ones(shape=(500,), dtype=bool)
X = nps.concatenate([X1, X2], axis=0)
y = nps.concatenate([y1, y2], axis=0)


# Train Logistic Regression Model.
model = LogisticRegression(solver="newton-cg", tol=1e-8, max_iter=1)
model.fit(X, y)
y_pred = model.predict(X)
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
