import time

import nums.numpy as nps


def test_permutation_subscript():
    samples, features = 10 ** 6, 30
    X = nps.random.rand(samples, features)
    idx = nps.random.permutation(X.shape[0]).astype(int)
    num_train = int(X.shape[0] * 0.8)
    num_test = X.shape[0] - num_train

    t = time.time()
    X_train, X_test = X[idx[:num_train]], X[idx[num_train:]]
    print("prep time", time.time() - t)
    X_train.touch()
    X_test.touch()
    print("exec time", time.time() - t)


if __name__ == "__main__":
    test_permutation_subscript()
