import numpy as np

class GeneralizedLinearRegressor(object):

    def __init__(self, learn_rate = .1, 
                       conv_thres = .0005,
                       method = 'grad'):
        self.conv_thres = conv_thres
        self.learn_rate = learn_rate
        self.coefs = None
        self.linkf = None

    def fit(self, X, y):
        self._init_coefs(X)
        dev, d_dev, dd_dev = self._deviance_derivs(X, y)
        improvement = float("inf")
        while improvement > self.conv_thres:
            print "Iter..."
            self._update_coefs(d_dev, dd_dev)
            new_dev, d_dev, dd_dev = self._deviance_derivs(X, y)
            improvement = np.abs(dev - new_dev)
            dev = new_dev

    def predict(self, X):
        return self.linkf(self._predict_lp(X))

    def _predict_lp(self, X):
        return np.dot(X, self.coefs) 

    def _init_coefs(self, X):
        n_coef = X.shape[1]
        self.coefs = np.zeros(n_coef)

    def _update_coefs(self, d_deviance, dd_deviance):
        adjustment = np.linalg.solve(dd_deviance, d_deviance)
        self.coefs = self.coefs - adjustment

    def _deviance_derivs(self, X, y):
        raise NotImplementedError("Deviance calculations must be implemented "
                                  "in a subclass of GeneralizedLinearRegressor"
              )
