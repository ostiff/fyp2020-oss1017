"""
IQR Transformer
===============
"""

# Sklearn libraries
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# --------------------------------------------------------------------------
#                       Inter-Quantile Range filter
# --------------------------------------------------------------------------
class IQRTransformer(BaseEstimator, TransformerMixin):
    """Description...

    .. note: Does it make to do it for different classes?
    .. note: __repr__ using scikits methods?
    """

    def __init__(self, iqrrange=[25, 75], coefficient=1.5):
        """The constructor"""
        self.iqrrange = iqrrange
        self.coefficient = coefficient
        self.lower_coefs_ = None
        self.upper_coefs_ = None

    def __repr__(self):
        """The representation"""
        return "IQRFilter(iqrrange=%s, coefficient=%s)" % \
               (self.iqrrange, self.coefficient)

    # --------------------------
    # helper methods
    # --------------------------
    def _fit(self, X, y=None):
        """This method computes the lower and upper percentiles
        """
        # Compute lower and uper quartiles
        lower_quartiles, upper_quartiles = \
            np.nanpercentile(X, self.iqrrange, axis=0)

        # Compute the interquantile range
        iqrs = (upper_quartiles - lower_quartiles) * self.coefficient

        # Set parameters
        return lower_quartiles - iqrs, upper_quartiles + iqrs

    def _transform(self, X, y=None):
        """This method filters single category.

        Parameters
        ----------
        X :

        Returns
        -------
        np.ndarray
        """
        # Copy X
        F = np.copy(X)

        # Indexes
        is_lower = F < self.lower_coefs_[0, :]
        is_upper = F > self.upper_coefs_[0, :]

        # Filter
        F[is_lower | is_upper] = np.nan

        # Return
        return F

    # --------------------------
    # main methods
    # --------------------------
    def fit(self, X, y=None):
        """This method fits single category.

        Parameters
        ----------
        X :

        Returns
        -------
        IQRFIlter instance
        """
        # Create the array coefficients
        self.lower_coefs_, self.upper_coefs_ = self._fit(X)

        # Format to array
        self.lower_coefs_ = self.lower_coefs_.reshape(1, -1)
        self.upper_coefs_ = self.upper_coefs_.reshape(1, -1)

        # Return
        return self

    def transform(self, X, y=None):
        """ This method...
        """
        # The object has not been previously fitted
        if self.lower_coefs_ is None or self.upper_coefs_ is None:
            raise TypeError("The instance IQRFilter has not been fitted.")

        # Return
        return self._transform(X, y)


if __name__ == '__main__':

    # Import
    import numpy as np
    import warnings

    # Import specific
    from sklearn.datasets import make_classification

    # ------------------------------------
    # basic configuration
    # ------------------------------------
    # Ignore all the warnings
    warnings.simplefilter('ignore')

    # Set print options
    np.set_printoptions(precision=2)

    # ------------------------------------
    # create data
    # ------------------------------------
    # Create feature data
    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 1, 2, 3],
                     [1, 2, 3, 4],
                     [1, 2, 3, 3],
                     [3, 7, 3, 4],
                     [1, 2, 3, 3],
                     [3, 7, 3, 4],
                     [1, 2, 3, 4],
                     [3, 6, 3, 4],
                     [2, 2, -55, 55]], np.float64)

    # Create categories
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # --------------
    # IQR filtering
    # --------------
    # Create filter object
    iqr = IQRTransformer(iqrrange=[25, 75], coefficient=1.5)

    # .. note: There is no need to pass y, it is just for
    #          compatibility with other transformers.

    # Fit and transform
    print("\nOriginal:")
    print(data)
    print("\nTransformed:")
    print(iqr.fit_transform(data))
    print("\nTransformed:")
    print(iqr.fit_transform(data, y))