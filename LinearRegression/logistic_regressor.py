import numpy as np
from glm import GeneralizedLinearRegressor

class LogisticRegressor(GeneralizedLinearRegressor):

    def __init__(self, learn_rate = .1, conv_thres = .0005):
        GeneralizedLinearRegressor.__init__(self, learn_rate, conv_thres)
        self.linkf = lambda x: 1/(1 + np.exp(-x)) 

    def _d_and_deviance(self, X, y):
        N = X.shape[0]
        p = self.predict(X)
        resids = y - p
        deviance = -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / N
        d_deviance = -2 * np.dot(X.T, y - p) / N
        return (deviance, d_deviance)
