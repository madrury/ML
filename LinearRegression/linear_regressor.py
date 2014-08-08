import numpy as np

class LinearRegressor(object):

    def __init__(self, learn_rate = .1, conv_thres = .0005):
        self.conv_thres = conv_thres
        self.learn_rate = learn_rate
        self.coefs = None

    def fit(self, X, y):
        self._init_coefs(X)
        dev, ddev = self._d_and_deviance(X, y)
        improvement = float("inf")
        while improvement > self.conv_thres:
            self._update_coefs(ddev)
            new_dev, ddev = self._d_and_deviance(X, y)
            improvement = np.abs(dev - new_dev)
            dev = new_dev

    def predict(self, X):
        return np.dot(X, self.coefs) 

    def _init_coefs(self, X):
        n_coef = X.shape[1]
        self.coefs = np.zeros(n_coef)

    def _d_and_deviance(self, X, y):
        preds = self.predict(X)
        resids = preds - y
        deviance = np.sum(resids ** 2) / X.shape[0]
        d_deviance = 2*np.dot(resids, X) / X.shape[0]
        return (deviance, d_deviance)

    def _update_coefs(self, d_deviance):
        self.coefs = self.coefs - self.learn_rate*d_deviance
