import numpy as np
from glm import GeneralizedLinearRegressor

class LinearRegressor(GeneralizedLinearRegressor):

    def __init__(self, learn_rate = .1, conv_thres = .0005):
        GeneralizedLinearRegressor.__init__(self, learn_rate, conv_threas)
        self.linkf = lambda x: x

    def _d_and_deviance(self, X, y):
        preds = self.predict(X)
        resids = preds - y
        deviance = np.sum(resids ** 2) / X.shape[0]
        d_deviance = 2*np.dot(resids, X) / X.shape[0]
        return (deviance, d_deviance)
