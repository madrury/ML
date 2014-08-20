import numpy as np
from glm import GeneralizedLinearRegressor

class LogisticRegressor(GeneralizedLinearRegressor):

    def __init__(self, conv_thres = .0005):
        GeneralizedLinearRegressor.__init__(self, conv_thres)
        self.linkf = lambda x: 1/(1 + np.exp(-x)) 

    def _deviance_derivs(self, X, y):
        N = X.shape[0]
        p = self.predict(X)
        resids = y - p
        deviance = -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / N
        d_deviance = -2 * np.dot(X.T, y - p) / N
        # This calculates PX, where P is the diagonal matrix consisting of 
        # p(1-p) values, without actually creating the diagonal matrix
        scaled_Xt = p * (1-p) * X.T
        dd_deviance = 2 * np.dot(scaled_Xt, X) / N
        return (deviance, d_deviance, dd_deviance)
