import logging

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class AbstractEPM(object):
    '''Abstract implementation of the EPM API 
        
        Note: The input dimensions of Y for training
        and the output dimensions of all predictions 
        (also called n_objectives)
        depends on the concrete implementation of this
        abstract class.
    '''

    def __init__(self,
                 instance_features: np.ndarray=None,
                 pca_components: float=None):
        '''
        Constructor
        
        Parameters
        ----------
        instance_features: np.ndarray (I, K)
            Contains the K dimensional instance features
            of the I different instances
        pca_components: float
            if set to a float, use PCA to reduce dimensionality of instance features
            also requires to set n_feats (> pca_dims)

        '''
        self.instance_features = instance_features
        self.pca_components = pca_components
        if instance_features is not None:
            self.n_feats = instance_features.shape[1]
        else:
            self.n_feats = 0

        self.n_params = None  # will be updated on train()

        self.pca = None
        self.scaler = None
        if self.pca_components and self.n_feats > self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            self.scaler = MinMaxScaler()

        # Never use a lower variance than this
        self.var_threshold = 10 ** -5

    def train(self, X, Y, **kwargs):
        '''Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            The corresponding target values. n_objectives must match the
            number of target names specified in the constructor.

        Returns
        -------
        self
        '''

        self.n_params = X.shape[1] - self.n_feats

        # reduce dimensionality of features of larger than PCA_DIM
        if self.pca:
            X_feats = X[:, -self.n_feats:]
            # scale features
            X_feats = self.scaler.fit_transform(X_feats)
            X_feats = np.nan_to_num(X_feats)  # if features with max == min
            # PCA
            X_feats = self.pca.fit_transform(X_feats)
            X = np.hstack((X[:, :self.n_params], X_feats))
            if hasattr(self, "types"):
                # for RF, adapt types list
                # if X_feats.shape[0] < self.pca, X_feats.shape[1] ==
                # X_feats.shape[0]
                self.types = np.array(np.hstack((self.types[:self.n_params], np.zeros((X_feats.shape[1])))),
                                      dtype=np.uint)
        return self._train(X, Y)

    def _train(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        '''Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            The corresponding target values. n_objectives must match the
            number of target names specified in the constructor.

        Returns
        -------
        self
        '''
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        '''
        Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        '''
        if self.pca:
            X_feats = X[:, -self.n_feats:]
            X_feats = self.scaler.transform(X_feats)
            X_feats = self.pca.transform(X_feats)
            X = np.hstack((X[:, :self.n_params], X_feats))

        return self._predict(X)

    def _predict(self, X: np.ndarray):
        '''
        Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        '''
        raise NotImplementedError()

    def predict_marginalized_over_instances(self, X):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean, var = self.predict(X)
            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        else:
            n_instances = len(self.instance_features)

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.bounds.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.bounds.shape[0],
                              X.shape[1]))

        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            X_ = np.hstack(
                (np.tile(x, (n_instances, 1)), self.instance_features))
            means, vars = self.predict(X_)
            # use only mean of variance and not the variance of the mean here
            # since we don't want to reason about the instance hardness distribution
            var_x = np.mean(vars)  # + np.var(means)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean[i] = np.mean(means)

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var
