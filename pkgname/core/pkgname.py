"""
Module to encapsulate all models using sklearn Mixins.
Make pipelineable models.
"""

import numpy as np
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from pkgname.core.AE.autoencoder import Autoencoder as base_autoencoder

class Autoencoder(BaseEstimator, TransformerMixin):
    """Wrapper for using pkgname.core.AE.autoencoder as sklearn Transformer"""

    def __init__(self, input_size=5, layers=None,
                 latent_dim=2, lr=0.0001, batch_size = 32,
                 device="cpu"):

        self.device = device
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.layers = layers if layers else []
        self.lr = lr
        self.batch_size = batch_size


    def fit(self, X, **fit_params):
        """Train the model with X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if self.latent_dim < 1:
            raise ValueError("latent_dim should be at least 1.")

        if self.input_size != X.shape[1]:
            raise ValueError("input_size should match n_features with "
                             "X : array-like of shape (n_samples, n_features).")


        self._model = base_autoencoder(input_size=self.input_size,
                                       layers=self.layers,
                                       latent_dim=self.latent_dim,
                                       device=self.device).to(self.device)

        return self

    def transform(self, X):
        """Apply the dimensionality reduction (encode) on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be encoded, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : ndarray of shape (n_samples, latent_dim)
            Transformed values.
        """
        return self._transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Train the model with X and apply the dimensionality reduction (encode) on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
        Returns
        -------
        X_new : ndarray of shape (n_samples, latent_dim)
            Transformed values.
        """
        return self.fit(X, **fit_params).transform(X)

    def _transform(self, X):

        loader_X = DataLoader(X, self.batch_size, shuffle=False)

        return self._model.encode_inputs(loader_X)


