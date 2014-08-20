import numpy as np
from glm import GeneralizedLinearRegressor

class LinearRegressor(GeneralizedLinearRegressor):

    def __init__(self, conv_threas = .0005):
        GeneralizedLinearRegressor.__init__(self, conv_threas)
        self.linkf = lambda x: x

    def _deviance_derivs(self, X, y):
        N = X.shape[0]
        preds = self.predict(X)
        resids = preds - y
        deviance = np.sum(resids ** 2) / N
        d_deviance = 2*np.dot(resids, X) / N
        dd_deviance = 2*np.dot(X.T, X) / N
        return (deviance, d_deviance, dd_deviance)
