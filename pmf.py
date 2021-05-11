import numpy as np

from nums.core.application_manager import instance as _instance

from sklearn.metrics import mean_squared_error
import nums.numpy as nps

class Factorization(object):

    def __init__(self, train_size=0.75, lambda_U = 0.3, lambda_V = 0.3):

        self._app = _instance()

        self.n_dims = 5
        self.parameters = {}

        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.n_users = 10
        self.n_movies = 10
       
        self.train_set = nps.random.randn_sparse(10, 10) 
        self.test_set = nps.random.randn_sparse(3, 3) 

        self.R = self.train_set
        self.U = nps.zeros((self.n_dims, self.n_users), dtype=np.float64)
        self.V = nps.random.randn(self.n_dims, self.n_movies)
    
    def update(self):
        for i in range(self.n_users):
            q = self.R[i, :] > 0
            V_j = self.V[:, q]

            Q = nps.matmul(V_j, V_j.T) + self.lambda_U * nps.identity(self.n_dims)
            QQ = _instance().inv(Q)

            Y = self.R[:, q][i, :]
            YY = nps.matmul(Y, V_j.T)

            self.U[:, i] = nps.matmul(QQ, YY)
    
    def predict(self, user_id, movie_id):
        u = self.U[:, user_id].T.reshape(1, -1)
        v = self.V[:, movie_id].reshape(-1, 1)
        
        r_ij = u @ v

        return r_ij[0][0]

    def evaluate(self, dataset):
        rows, cols, vals = dataset.find(self._app)
        
        u = self.U[:, rows]
        v = self.V[:, cols]
        predictions = u * v
        q = nps.sum(predictions, axis=0)

        return mean_squared_error(q.get(), vals.get(), squared=False)

    def fit(self):
        log_aps = []
        rmse_train = []
        rmse_test = []

        rmse_train.append(self.evaluate(self.train_set))
        rmse_test.append(self.evaluate(self.test_set))
        
        # for k in range(self.n_epochs):
        #     self.update()

        #     rmse_train.append(self.evaluate(self.train_set))
        #     rmse_test.append(self.evaluate(self.test_set))
            
        #     # log_ap = log_a_posteriori()
        #     # log_aps.append(log_ap)

        print(rmse_train)
        print(rmse_test)

f = Factorization()
f.fit()




