"""
Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.8
"""

import pandas as pd
import numpy  as np

import warnings
import math
import re

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin														
														
from sklearn.utils            import check_X_y
from sklearn.utils            import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing    import KBinsDiscretizer
from sklearn.preprocessing    import LabelEncoder
from sklearn.preprocessing    import MinMaxScaler
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.cluster          import KMeans

from scipy.optimize import differential_evolution

# Compressors

import bz2
import lzma
import zlib

# Supported classifiers

from sklearn.naive_bayes    import MultinomialNB
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm			import SVC

# Supported regressors

from sklearn.linear_model   import LinearRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.svm            import LinearSVR
from sklearn.neural_network import MLPRegressor

# Supported time series
# 
# - Autoregression
# - Moving Average
# - Simple Exponential Smoothing

#
# Private Helper Functions
#

"""
Count the number of occurences of a discretized 1d or 2d space
for classification or regression problems
    
Parameters
----------
x1, x2: array-like, shape (n_samples)
numeric1, numeric2: if the variable is numeric or not
       
Returns
-------
A vector with the frequencies of the unique values computed.
"""
def _unique_count(x1, numeric1, x2=None, numeric2=None):

    # Process first variable

    if not numeric1:

        # Econde categorical values as numbers
        le = LabelEncoder()
        le.fit(x1)
        x1 = le.transform(x1)

    else:

        # Discretize variable
        if x2 is not None:
            x1 = _discretize_vector(x1, dim=1)
        else:
            x1 = _discretize_vector(x1, dim=2)            

    # Process second variable

    if x2 is not None:

        if not numeric2:

            # Econde categorical values as numbers
            le = LabelEncoder()
            le.fit(x2)
            x2 = le.transform(x2)

        else:

            # Discretize variable
            x2 = _discretize_vector(x2, dim=2)

        x = (x1 + x2) * (x1 + x2 + 1) / 2 + x2
        x = x.astype(int)

    else:
        
        x = x1

    # Return count        
    
    y     = np.bincount(x)
    ii    = np.nonzero(y)[0]
    count = y[ii]

    return count


"""
Discretize a continous variable using a "uniform" strategy
    
Parameters
----------
x  : array-like, shape (n_samples)
dim: Number of dimensions of space
       
Returns
-------
A new discretized vector of integers.
"""
def _discretize_vector(x, dim=1):

    length = x.shape[0]
    new_x  = x.copy().reshape(-1, 1)

    # TODO: Think about this
    # Optimal number of bins
    optimal_bins = int(np.cbrt(length))

    # TODO: Think about this
    # if dim == 1:
    #     optimal_bins = int(np.sqrt(length))
    # else:
    #     optimal_bins = int(np.sqrt(np.sqrt(length)))
    
    # Correct the number of bins if it is too small
    if optimal_bins <= 1:
        optimal_bins = 2
    
    # Repeat the process until we have data in all the intervals

    total_bins    = optimal_bins
    previous_bins = 0
    stop          = False

    while stop == False:

        # Avoid those annoying warnings
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")

            est = KBinsDiscretizer(n_bins=total_bins, encode='ordinal', strategy="uniform")
            est.fit(new_x)
            tmp_x = est.transform(new_x)[:,0].astype(dtype=int)

        y = np.bincount(tmp_x)
        actual_bins = len(np.nonzero(y)[0])

        if previous_bins == actual_bins:
            # Nothing changed, better stop here
            stop = True

        if actual_bins < optimal_bins:
            # Too few intervals with data
            previous_bins = actual_bins
            add_bins      = int( np.round( (length * (1 - actual_bins / optimal_bins)) / optimal_bins ) )
            total_bins    = total_bins + add_bins
        else:
            # All intervals have data
            stop = True

    new_x = est.transform(new_x)[:,0].astype(dtype=int)

    return new_x


"""
Compute the length of a list of features (1d or 2d)
and / or a target variable (classification or regression)
using an optimal code using the frequencies of the categorical variables
or a discretized version of the continuous variables
    
Parameters
----------
x1, x2: array-like, shape (n_samples)
numeric1, numeric2: if the variable is numeric or not
       
Returns
-------
Return the length of the encoded dataset (float)
"""
def _optimal_code_length(x1, numeric1, x2=None, numeric2=None):

    count = _unique_count(x1=x1, numeric1=numeric1, x2=x2, numeric2=numeric2)
    ldm = np.sum(count * ( - np.log2(count / len(x1) )))
    
    return ldm


#
# Class Miscoding
# 
class Miscoding(BaseEstimator):
    """
    Given a dataset X = {x1, ..., xp} composed by p features, and a target
    variable y, the miscoding of the feature xj measures how difficult is to
    reconstruct y given xj, and the other way around. We are not only
    interested in to identify how much information xj contains about y, but
    also if xj contains additional information that is not related
    to y (which is a bad thing). Miscoding also takes into account that
    feature xi might be redundant with respect to feature xj.

    The fastautoml.Miscoding class allow us to compute the relevance of
    features, the quality of a dataset, and select the optimal subset of
    features to include in a study

    Example of usage:
        
        from fastautoml.fastautoml import Miscoding
        miscoding = Miscoding()
        miscoding.fit(X, y)
        msd = miscoding.miscoding_features()
    """

    def __init__(self, X_type="numeric", y_type="numeric", redundancy=False):
        """
        Initialization of the class Miscoding
        
        Parameters
        ----------
        X_type:     The type of the features, numeric, mixed or categorical
        y_type:     The type of the target, numeric or categorical
        redundancy: if "True" takes into account the redundancy between features
                    to compute the miscoding, if "False" only the miscoding with
                    respect to the target variable is computed.
          
        """        

        valid_X_types = ("numeric", "mixed", "categorical")
        valid_y_types = ("numeric", "categorical")

        if X_type not in valid_X_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_X_types, X_type))

        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        self.X_type     = X_type
        self.y_type     = y_type
        self.redundancy = redundancy
        
        return None
    
    
    def fit(self, X, y):
        """
        Learn empirically the miscoding of the features of X
        as a representation of y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            array-like, numpy or pandas array in case of numerical types
            pandas array in case of mixed or caregorical types
            
        y : array-like, shape (n_samples)
            The target values as numbers or strings.
            
        Returns
        -------
        self
        """

        if self.X_type == "mixed" or self.X_type == "categorical":

            if isinstance(X, pd.DataFrame):
                self.X_isnumeric = [np.issubdtype(my_type, np.number) for my_type in X.dtypes]
            else:
                raise ValueError("Only DataFrame is allowed for X of type 'mixed' and 'categorical."
                                 "Got type {!r} instead."
                                 .format(type(X)))
                
        else:
            self.X_isnumeric = [True] * X.shape[1]

        if self.y_type == "numeric":
            self.y_isnumeric = True
        else:
            self.y_isnumeric = False
        
        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        # TODO: Joint miscoding is not supported yet
        self.regular_ = self._miscoding_features_single()

        # TODO: Fix!
        # if self.redundancy:
        #     self.regular_ = self._miscoding_features_joint()
        # else:
        #     self.regular_ = self._miscoding_features_single()

        self.adjusted_ = 1 - self.regular_

        if np.sum(self.adjusted_) != 0:
            self.adjusted_ = self.adjusted_ / np.sum(self.adjusted_)

        if np.sum(self.regular_) != 0:
            self.partial_  = self.adjusted_ - self.regular_ / np.sum(self.regular_)
        else:
            self.partial_  = self.adjusted_
        
        return self


    def miscoding_features(self, mode='adjusted'):
        """
        Return the miscoding of the target given the features

        Parameters
        ----------
        mode  : the mode of miscoding, possible values are 'regular' for
                the true miscoding, 'adjusted' for the normalized inverted
                values, and 'partial' with positive and negative
                contributions to dataset miscoding.
            
        Returns
        -------
        Return a numpy array with the miscodings
        """
        
        check_is_fitted(self)
        
        if mode == 'regular':
            return self.regular_
        elif mode == 'adjusted':
            return self.adjusted_
        elif mode == 'partial':
            return self.partial_
        else:
            valid_modes = ('regular', 'adjusted', 'partial')
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

        return None 

    
    def cross_miscoding(self, attribute, min_lag=0, max_lag=None, mode='adjusted'):

        check_is_fitted(self)

        valid_modes = ('regular', 'adjusted', 'partial')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))
        
        lag_mscd = list()
        
        # Use a default value for those lazy programmers
        if max_lag == None:
            max_lag = int(np.sqrt(self.X_.shape[0]))

        for i in np.arange(start=min_lag, stop=max_lag):

            # Compute lagged vectors
            new_y = self.y_.copy()
            new_y = np.roll(new_y, -i)
            new_y = new_y[:-i]

            new_x = self.X_[:,attribute].copy()
            new_x = new_x[:-i]

            ldm_y  = _optimal_code_length(x1=new_y, numeric1=self.y_isnumeric)
            ldm_X  = _optimal_code_length(x1=new_x, numeric1=self.X_isnumeric[attribute])
            ldm_Xy = _optimal_code_length(x1=new_x, numeric1=self.X_isnumeric[attribute], x2=new_y, numeric2=self.y_isnumeric)
                       
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            lag_mscd.append(mscd)
                
        regular = np.array(lag_mscd)

        if mode == 'regular':

            return regular

        elif mode == 'adjusted':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)
            
            return adjusted

        elif mode == 'partial':

            if np.sum(regular) != 0:
                partial  = adjusted - regular / np.sum(regular)
            else:
                partial  = adjusted

            return partial

        else:

            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))


    def miscoding_model(self, model):
        """
        Compute the partial joint miscoding of the dataset used by a model
        
        Parameters
        ----------
        model : a model of one of the supported classes
                    
        Returns
        -------
        Return the miscoding (float)
        """

        check_is_fitted(self)

        if isinstance(model, MultinomialNB):
            subset = self._MultinomialNB(model)
        elif isinstance(model, DecisionTreeClassifier):
            subset = self._DecisionTreeClassifier(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='linear':
            subset = self._LinearSVC(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='poly':
            subset = self._SVC(model)
        elif isinstance(model, MLPClassifier):
            subset = self._MLPClassifier(model)
        elif isinstance(model, LinearRegression):
            subset = self._LinearRegression(model)
        elif isinstance(model, DecisionTreeRegressor):
            subset = self._DecisionTreeRegressor(model)
        elif isinstance(model, LinearSVR):
            subset = self._LinearSVR(model)
        elif isinstance(model, MLPRegressor):
            subset = self._MLPRegressor(model)            
        else:
            # Rise exception
            raise NotImplementedError('Model {!r} not supported'
                                     .format(type(model)))

        return self.miscoding_subset(subset)
        

    def miscoding_subset(self, subset):
        """
        Compute the partial joint miscoding of a subset of the features
        
        Parameters
        ----------
        subset : array-like, shape (n_features)
                 1 if the attribute is in use, 0 otherwise
        
        Returns
        -------
        Return the miscoding (float)
        """

        check_is_fitted(self)

        # Avoid miscoding greater than 1
        top_mscd = 1 + np.sum(self.partial_[self.partial_ < 0])
        miscoding = top_mscd - np.dot(subset, self.partial_)
                
        # Avoid miscoding smaller than zero
        if miscoding < 0:
            miscoding = 0

        return miscoding


    def features_matrix(self, mode="adjusted"):
        """
        Compute a matrix of adjusted miscodings for the features

        Parameters
        ----------
        mode  : the mode of miscoding, possible values are 'regular' for the true miscoding
                and 'adjusted' for the normalized inverted values

        Returns
        -------
        Return the matrix (n_features x n_features) with the miscodings (float)
        """

        check_is_fitted(self)
        
        valid_modes = ('regular', 'adjusted')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

        miscoding = np.zeros([self.X_.shape[1], self.X_.shape[1]])

        # Compute the regular matrix

        for i in np.arange(self.X_.shape[1]-1):
            
            ldm_X1 = _optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])

            for j in np.arange(i+1, self.X_.shape[1]):

                ldm_X2   = _optimal_code_length(x1=self.X_[:,j], numeric1=self.X_isnumeric[j])
                ldm_X1X2 = _optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.X_[:,j], numeric2=self.X_isnumeric[j])
                       
                mscd = ( ldm_X1X2 - min(ldm_X1, ldm_X2) ) / max(ldm_X1, ldm_X2)
                
                miscoding[i, j] = mscd
                miscoding[j, i] = mscd

        if mode == "regular":
            return miscoding
                
        # Compute the normalized matrix
        
        normalized = np.zeros([self.X_.shape[1], self.X_.shape[1]])
        
        for i in np.arange(self.X_.shape[1]):

            normalized[i,:] = 1 - miscoding[i,:]
            normalized[i,:] = normalized[i,:] / np.sum(normalized[i,:])

        return normalized


    """
    Return the regular miscoding of the target given the features
            
    Returns
    -------
    Return a numpy array with the regular miscodings
    """
    def _miscoding_features_single(self):
                 
        miscoding = list()
                
        ldm_y = _optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

        for i in np.arange(self.X_.shape[1]):
                        
            ldm_X  = _optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])
            ldm_Xy = _optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.y_, numeric2=self.y_isnumeric)
                       
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            miscoding.append(mscd)
                
        miscoding = np.array(miscoding)

        return miscoding


    """
    Return the joint regular miscoding of the target given pairs features
            
    Returns
    -------
    Return a numpy array with the regular miscodings
    """
    # TODO: Not supported yet! Review.
    def _miscoding_features_joint(self):

        # Compute non-redundant miscoding
        mscd = self._miscoding_features_single()

        if self.X_.shape[1] == 1:
            # With one single attribute we cannot compute the joint miscoding
            return mscd

        #
        # Compute the joint miscoding matrix
        #         
               
        red_matrix = np.ones([self.X_.shape[1], self.X_.shape[1]])

        ldm_y = _optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

        for i in np.arange(self.X_.shape[1]-1):
                        
            for j in np.arange(i+1, self.X_.shape[1]):
                
                ldm_X1X2  = _optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.X_[:,j], numeric2=self.X_isnumeric[j])
                ldm_X1X2Y = _optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.y_, numeric2=self.y_isnumeric)
                
                tmp = ( ldm_X1X2Y - min(ldm_X1X2, ldm_y) ) / max(ldm_X1X2, ldm_y)
                                
                red_matrix[i, j] = tmp
                red_matrix[j, i] = tmp
        
        #
        # Compute the joint miscoding 
        #

        viu       = np.zeros(self.X_.shape[1], dtype=np.int8)
        miscoding = np.zeros(self.X_.shape[1])

        # Select the first two variables with smaller joint miscoding
        
        loc1, loc2 = np.unravel_index(np.argmin(red_matrix, axis=None), red_matrix.shape)
        jmscd1 = jmscd2 = red_matrix[loc1, loc2]

        # Avoid the case of selecting a member of the main diagonal
        if loc1 == loc2:
            loc2 = loc2 + 1
        
        viu[loc1] = 1
        viu[loc2] = 1

        # Scale down one of them
                
        tmp1 = mscd[loc1]
        tmp2 = mscd[loc2]
        
        if tmp1 < tmp2:
            jmscd1 = jmscd1 * tmp1 / tmp2
        elif tmp1 > tmp2:
            jmscd2 = jmscd2 * tmp2 / tmp1
        
        miscoding[loc1] = jmscd1
        miscoding[loc2] = jmscd2
 
        # Iterate over the number of features
        
        tmp = np.ones(self.X_.shape[1]) * np.inf
        
        for i in np.arange(2, self.X_.shape[1]):

            for j in np.arange(self.X_.shape[1]):
            
                if viu[j] == 1:
                    continue

                tmp[j] = (1 / np.sum(viu)) * np.sum(red_matrix[np.where(viu == 1), j])
                            
            viu[np.argmin(tmp)] = 1
            miscoding[np.argmin(tmp)] = np.min(tmp)

            tmp = np.ones(self.X_.shape[1]) * np.inf
        
        return miscoding


    """
    Compute the attributes in use for a multinomial naive Bayes classifier
    
    Return array with the attributes in use
    """
    def _MultinomialNB(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use
    

    """
    Compute the attributes in use for a decision tree
    
    Return array with the attributes in use
    """
    def _DecisionTreeClassifier(self, estimator):

        attr_in_use = np.zeros(self.X_.shape[1], dtype=int)
        features = set(estimator.tree_.feature[estimator.tree_.feature >= 0])
        for i in features:
            attr_in_use[i] = 1
            
        return attr_in_use


    """
    Compute the attributes in use for a linear support vector classifier
    
    Return array with the attributes in use
    """
    def _LinearSVC(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a support vector classifier with a polynomial kernel
    
    Return array with the attributes in use
    """
    def _SVC(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a multilayer perceptron classifier
    
    Return array with the attributes in use
    """
    def _MLPClassifier(self, estimator):

        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a linear regression
    
    Return array with the attributes in use
    """
    def _LinearRegression(self, estimator):
        
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a decision tree regressor
    
    Return array with the attributes in use
    """
    def _DecisionTreeRegressor(self, estimator):
        
        attr_in_use = np.zeros(self.X_.shape[1], dtype=int)
        features = set(estimator.tree_.feature[estimator.tree_.feature >= 0])
        for i in features:
            attr_in_use[i] = 1
            
        return attr_in_use


    """
    Compute the attributes in use for a linear support vector regressor
    
    Return array with the attributes in use
    """
    def _LinearSVR(self, estimator):

        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use

    """
    Compute the attributes in use for a multilayer perceptron regressor
    
    Return array with the attributes in use
    """
    def _MLPRegressor(self, estimator):

        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


#
# Class Inaccuracy
# 
class Inaccuracy(BaseEstimator):
    """
    The fastautoml.Inaccuracy class allow us to compute the quality of
    the predictions made by a trained model.

    Example of usage:
        
        from fastautoml.fastautoml import Inaccuracy
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)

        tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
        tree.fit(X, y)

        inacc = Inaccuracy()
        inacc.fit(X, y)
        inacc.inaccuracy_model(tree)

    """    

    def __init__(self, y_type="numeric"):
        """
        Initialization of the class Inaccuracy
        
        Parameters
        ----------
        y_type:     The type of the target, numeric or categorical
        """        

        valid_y_types = ("numeric", "categorical")

        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        self.y_type = y_type

        if y_type == "numeric":
            self.y_isnumeric = True
        else:
            self.y_isnumeric = False

        return None
    
    
    def fit(self, X, y):
        """
        Fit the inaccuracy with a dataset
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which models have been trained.
            
        y : array-like, shape (n_samples)
            Continuous and categorical variables are supported
            if the trained model support them.
            
        Returns
        -------
        self
        """
        
        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        self.y_ = np.array(self.y_)
                
        self.len_y = _optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)
        
        return self


    def inaccuracy_model(self, model):
        """
        Compute the inaccuracy of a model

        Parameters
        ----------       
        model : a trained model with a predict() method

        Returns
        -------         
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self)
        
        Pred      = model.predict(self.X_)
        len_pred  = _optimal_code_length(x1=Pred, numeric1=self.y_isnumeric)
        len_joint = _optimal_code_length(x1=Pred, numeric1=self.y_isnumeric, x2=self.y_, numeric2=self.y_isnumeric)
        inacc     = ( len_joint - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc 

    
    def inaccuracy_predictions(self, predictions):
        """
        Compute the inaccuracy of a list of predicted values

        Parameters
        ----------       
        pred : array-like, shape (n_samples)
               The list of predicted values

        Returns
        -------                
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self)

        pred = np.array(predictions)
        
        len_pred  = _optimal_code_length(x1=pred, numeric1=self.y_isnumeric)
        len_joint = _optimal_code_length(x1=pred, numeric1=self.y_isnumeric, x2=self.y_, numeric2=self.y_isnumeric)
        inacc     = ( len_joint - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc    


#
# Class Surfeit
# 
    
class Surfeit(BaseEstimator):

    def __init__(self, y_type="numeric", compressor="bz2"):
        """
        Initialization of the class Surfeit

        Parameters
        ----------
        y_type:     The type of the target, numeric or categorical
        compressor: The compressor used to encode the model

        Returns
        -------
        self
        """
	        
        if y_type == "numeric":
            self.y_isnumeric = True
        else:
            self.y_isnumeric = False

        self.y_type      = y_type   
        self.compressor  = compressor
        
        return None
    

    def fit(self, X, y):
        """Initialize the Surfeit class with dataset
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which models have been trained.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as integers or strings.
            
        Returns
        -------
        self
        """
        
        self.X_, self.y_ = check_X_y(X, y, dtype=None)
                
        self.len_y_ = _optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)
        
        return self
    

    def surfeit_model(self, model):
        """
        Compute the redundancy of a model

        Parameters
        ----------
        model : a model of one of the supported classeses
        
        Supported classes:
            MultinomialNB
            DecisionTreeClassifier
            MLPClassifier
            
        Returns
        -------
        Redundancy (float) of the model
        """

        if isinstance(model, MultinomialNB):
            model_str = self._MultinomialNB(model)
        elif isinstance(model, DecisionTreeClassifier):
            model_str = self._DecisionTreeClassifier(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='linear':
            model_str = self._LinearSVC(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='poly':
            model_str = self._SVC(model)
        elif isinstance(model, MLPClassifier):
            model_str = self._MLPClassifier(model)
        elif isinstance(model, LinearRegression):
            model_str = self._LinearRegression(model)
        elif isinstance(model, DecisionTreeRegressor):
            model_str = self._DecisionTreeRegressor(model)
        elif isinstance(model, LinearSVR):
            model_str = self._LinearSVR(model)
        elif isinstance(model, MLPRegressor):
            model_str = self._MLPRegressor(model)
        else:
            # Rise exception
            raise NotImplementedError('Model {!r} not supported'
                                     .format(type(model)))

        return self.surfeit_string(model_str)
        

    def surfeit_string(self, model_string):
        """
        Compute the redundancy of a model given as a string

        Parameters
        ----------
        model : a string based representation of the model
            
        Returns
        -------
        Redundancy (float) of the model
        """
    
        # Compute the model string and its compressed version
        emodel = model_string.encode()
        
        if self.compressor == "lzma":
            compressed = lzma.compress(emodel, preset=9)
        elif self.compressor == "zlib":
            compressed = zlib.compress(emodel, level=9)
        else: # By default use bz2
            compressed = bz2.compress(emodel, compresslevel=9)
        
        km = len(compressed)
        lm = len(emodel)

        # Check if the model is too small to compress        
        if km > lm:
            return 1 - 3/4    # Experimental value

        if self.len_y_ < km:
            # redundancy = 1 - l(C(y)) / l(m)
            redundancy = 1 - self.len_y_ / lm
        else:
            # redundancy = 1 - l(m*) / l(m)
            redundancy = 1 - km / lm
                            
        return redundancy


    """
    Convert a MultinomialNB classifier into a string
    """
    def _MultinomialNB(self, estimator):
        #
        # Discretize probabilities
        #

        py    = _discretize_vector(np.exp(estimator.class_log_prior_))
        
        theta = np.exp(estimator.feature_log_prob_)
        theta = theta.flatten()
        theta = _discretize_vector(theta)
        theta = np.array(theta)
        theta = theta.reshape(estimator.feature_log_prob_.shape)
        
        #
        # Create the model
        #

        # Header
        string = "def Bayes(X):\n"
 
        # Target probabilities
        string = string + "    Py["
        for i in np.arange(len(py)):
            string = string + str(py) + ", "
        string = string + "]\n"
            
        # Conditional probabilities
        string = string + "    theta["        
        for i in np.arange(len(theta)):
            string = string + str(theta[i]) + ", "
        string = string + "]\n"

        string = string + "    y_hat    = None\n"
        string = string + "    max_prob = 0\n"
        string = string + "    for i in range(len(estimator.classes_)):\n"
        string = string + "        prob = 1\n"
        string = string + "        for j in range(len(theta[i])):\n"
        string = string + "            prob = prob * theta[i][j]\n"
        string = string + "        prob = py[i] *  prob\n"
        string = string + "        if prob > max_prob:\n"
        string = string + "            y_hat = estimator.classes_[i]\n"
        string = string + "    return y_hat\n"
                
        return string
    
    
    """
    Convert a LinearSVC classifier into a string
    """
    def _LinearSVC(self, estimator):
  
        #
        # Discretize similarities
        #

        M = estimator.coef_
        shape = M.shape
        M = M.flatten()
        M = _discretize_vector(M)
        M = np.array(M)
        M = M.reshape(shape)
        
        intercept = estimator.intercept_
        shape = intercept.shape
        intercept = intercept.flatten()
        intercept = _discretize_vector(intercept)
        intercept = np.array(intercept)
        intercept = intercept.reshape(shape)
        
        classes = estimator.classes_
        
        if len(classes) == 2:
        
            #
            # Create the model
            #

            # Header
            string = "def LinearSVC(X):\n"
                 
            # Similarities
            string = string + "    M = ["        
            for j in np.arange(len(M)-1):
                string = string + str(M[j]) + ", "
            string = string + str(M[-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            string = string + str(intercept)
            string = string + "]\n"
            
            # Computation of the decision function
            string = string + '    y_hat = [None]*len(X)'
            string = string + '    for i in range(len(X)):\n'
            string = string + '        prob = 0\n'
            string = string + '        for k in range(len(M)):\n'
            string = string + '            prob = prob + X[i][k] * M[k]\n'
            string = string + '        prob = prob + intercept\n'
			
            #Prediction
            string = string + '        if prob > 0:\n'
            string = string + '            y_hat[i] = 0\n'
            string = string + '        else:\n'
            string = string + '            y_hat[i] = 1\n'
            string = string + '    return y_hat\n'
        
        else:
        
            #
            # Create the model
            #

            # Header
            string = "def LinearSVC(X):\n"
                 
            # Similarities
            string = string + "    M = ["        
            for i in np.arange(len(M)-1):
                string = string + "["
                for j in np.arange(len(M[i])-1):
                    string = string + str(M[i][j]) + ", "
                string = string + str(M[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(M[-1])-1):
                string = string + str(M[-1][j]) + ", "
            string = string + str(M[-1][-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            for i in np.arange(len(intercept)-1):
                string = string + str(intercept[i])
                string = string + ", "
            string = string + str(intercept[-1])
            string = string + "]\n"
            
            string = string + "    classes = ["        
            for i in np.arange(len(classes-1)):
                string = string + str(classes[i]) + ", "
            string = string + str(classes[-1])
            string = string + "]]\n"
           
		    # Computation of the decision function ('ovo' strategy)
            string = string + '    y_hat = [None]*len(X)'
            string = string + '    for i in range(len(X)):\n'
            string = string + '        votes = [0]*len(classes)\n'
            string = string + '        idx = 0\n'
            string = string + '        for j in range(len(classes)):\n'
            string = string + '            for l in range(len(classes)-j-1):\n'
            string = string + '                prob = 0\n'
            string = string + '                for k in range(len(M[idx])):\n'
            string = string + '                    prob = prob + X[i][k] * M[idx][k]\n'
            string = string + '                prob = prob + intercept[idx]\n'
            string = string + '                if prob > 0:\n'
            string = string + '                    votes[j] = votes[j] + 1\n'
            string = string + '                else:\n'
            string = string + '                    votes[l+j+1] = votes[l+j+1] + 1\n'
            string = string + '                idx = idx + 1\n'
            
            # Prediction
            string = string + '        max_vote = 0\n'
            string = string + '        i_max_vote = 0\n'
            string = string + '        for k in range(len(votes)):\n'
            string = string + '            if votes[k]>max_vote:\n'
            string = string + '                max_vote = votes[k]\n'
            string = string + '                i_max_vote = k\n'
            string = string + '        y_hat[i] = classes[i_max_vote]\n'
            string = string + '    return y_hat\n'
        
        return string


    """
	Convert a SVC classifier into a string
	"""
    string = ""
    affiche = 1
    def _SVC(self, estimator):
	
        #
        # Discretize similarities
        #


        M = estimator._dual_coef_
        shape = M.shape
        M = M.flatten()
        M = _discretize_vector(M)
        M = np.array(M)
        M = M.reshape(shape)
        
        support_vectors = estimator.support_vectors_
        shape = support_vectors.shape
        support_vectors = support_vectors.flatten()
        support_vectors = _discretize_vector(support_vectors)
        support_vectors = np.array(support_vectors)
        support_vectors = support_vectors.reshape(shape)
        
        if len(estimator.classes_) == 2:
        
            #
            # Create the model

            # Header
            string = "def SVC(X):\n"
                 
            # Similarities
            string = string + "    dual_coef = ["        
            for i in np.arange(len(M)-1):
                string = string + "["
                for j in np.arange(len(M[i])-1):
                    string = string + str(M[i][j]) + ", "
                string = string + str(M[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(M[-1])-1):
                string = string + str(M[-1][j]) + ", "
            string = string + str(M[-1][-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            string = string + str(estimator.intercept_)
            string = string + "]\n"
            
            string = string + "    classes = ["        
            for i in np.arange(len(estimator.classes_)-1):
                string = string + str(estimator.classes_[i]) + ", "
            string = string + str(estimator.classes_[-1])
            string = string + "]\n"
            
            string = string + "    support_vectors = ["        
            for i in np.arange(len(support_vectors)-1):
                string = string + "["
                for j in np.arange(len(support_vectors[i])-1):
                    string = string + str(support_vectors[i][j]) + ", "
                string = string + str(support_vectors[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(support_vectors[-1])-1):
                string = string + str(support_vectors[-1][j]) + ", "
            string = string + str(support_vectors[-1][-1])
            string = string + "]]\n"
            
            string = string + "    n_support = ["        
            for i in np.arange(len(estimator.n_support_)-1):
                string = string + str(estimator.n_support_[i]) + ", "
            string = string + str(estimator.n_support_[-1])
            string = string + "]\n"
            
            string = string + "    degree = "
            string = string + str(estimator.degree)
            string = string + "    \n"
            
            string = string + "    gamma = "
            if estimator.gamma == 'scale':
                string = string + str(1/(len(support_vectors[0])*np.var(self.X_)))
            elif estimator.gamma == 'auto':
                string = string + str(1/len(support_vectors[0]))
            else:
                string = string + str(estimator.gamma)
            string = string + "    \n"
            
            string = string + "    r = "
            string = string + str(estimator.coef0)
            string = string + "    \n"

            # Computation of the decision function ('ovo' strategy)
            string = string + "    y_hat    = [None]*len(X)\n"
            string = string + "    for i in range(len(X)):\n"
            string = string + "        prob = 0\n" 
            string = string + "        for i_sv in range(len(support_vectors)):\n"
            string = string + "            sum = 0\n" 
            string = string + "            for k in range(len(X[i])):\n"
            string = string + "                sum = sum + support_vectors[i_sv][k] * X[i][k]\n"
            string = string + "            x = 1\n"
            string = string + "            for k in range(degree):\n"
            string = string + "                x = x * (gamma * sum + r)\n"
            string = string + "            prob = prob + x * dual_coef[i_sv]\n"
            string = string + "        prob = prob + intercept[0]\n"
	
            # Prediction
            string = string + "        if prob > 0:\n"
            string = string + "            y_hat[i] = 0\n"
            string = string + "        else:\n"
            string = string + "            y_hat[i] = 1\n"
            string = string + "    return y_hat\n"

        else:
        
            #
            # Create the model
            #

            # Header
            string = "def SVC(X):\n"
                 
            # Similarities
            string = string + "    dual_coef = ["        
            for i in np.arange(len(M)-1):
                string = string + "["
                for j in np.arange(len(M[i])-1):
                    string = string + str(M[i][j]) + ", "
                string = string + str(M[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(M[-1])-1):
                string = string + str(M[-1][j]) + ", "
            string = string + str(M[-1][-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            for i in np.arange(len(estimator.intercept_)-1):
                string = string + str(estimator.intercept_[i]) + ", "
            string = string + str(estimator.intercept_[-1])
            string = string + "]\n"
            
            string = string + "    classes = ["        
            for i in np.arange(len(estimator.classes_)-1):
                string = string + str(estimator.classes_[i]) + ", "
            string = string + str(estimator.classes_[-1])
            string = string + "]\n"
            
            string = string + "    support_vectors = ["        
            for i in np.arange(len(support_vectors)-1):
                string = string + "["
                for j in np.arange(len(support_vectors[i])-1):
                    string = string + str(support_vectors[i][j]) + ", "
                string = string + str(support_vectors[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(support_vectors[-1])-1):
                string = string + str(support_vectors[-1][j]) + ", "
            string = string + str(support_vectors[-1][-1])
            string = string + "]]\n"
            
            string = string + "    n_support = ["        
            for i in np.arange(len(estimator.n_support_)-1):
                string = string + str(estimator.n_support_[i]) + ", "
            string = string + str(estimator.n_support_[-1])
            string = string + "]\n"
            
            string = string + "    idx_support = ["        
            for i in np.arange(len(estimator.n_support_)):
                string = string + str(np.sum(estimator.n_support_[:i])) + ", "
            string = string + str(np.sum(estimator.n_support_))
            string = string + "]\n"
            
            string = string + "    degree = "
            string = string + str(estimator.degree)
            string = string + "    \n"
            
            string = string + "    gamma = "
            if estimator.gamma == 'scale':
                string = string + str(1/(len(support_vectors[0])*np.var(self.X_)))
            elif estimator.gamma == 'auto':
                string = string + str(1/len(support_vectors[0]))
            else:
                string = string + str(estimator.gamma)
            string = string + "    \n"
            
            string = string + "    r = "
            string = string + str(estimator.coef0)
            string = string + "    \n"

            # Computation of the decision function ('ovo' strategy)
            string = string + "    y_hat    = [None]*len(X)\n"
            string = string + "    for i in range(len(X)):\n"
            string = string + "        votes = [0]*len(classes)\n"
            string = string + "        idx = 0\n"
            string = string + "        for j in range(len(classes)):\n"
            string = string + "            for l in range(len(classes)-j-1):\n"
            string = string + "                prob = 0\n" 
            string = string + "                sum = 0\n" 
            string = string + "                for i_sv in range(idx_support[j],idx_support[j+1]):\n"
            string = string + "                    for k in range(len(X[i])):\n"
            string = string + "                        sum = sum + support_vectors[i_sv][k] * X[i][k]\n"
            string = string + "                    x = 1\n"
            string = string + "                    for k in range(degree):\n"
            string = string + "                        x = x * (gamma * sum + r)\n"
            string = string + "                    prob = prob + x * dual_coef[l+j][i_sv]\n"
            string = string + "                sum = 0\n"
            string = string + "                for i_sv in range(idx_support[j+l],idx_support[j+l+1]):\n"
            string = string + "                    for k in range(len(X[i])):\n"
            string = string + "                        sum = sum + support_vectors[i_sv][k] * X[i][k]\n"
            string = string + "                    x = 1\n"
            string = string + "                    for k in range(degree):\n"
            string = string + "                        x = x * (gamma * sum + r)\n"
            string = string + "                    prob = prob + x * dual_coef[j][i_sv]\n"
            string = string + "                prob = prob + intercept[idx]\n"
            string = string + "                if prob > 0:\n"
            string = string + "                    votes[j] = votes[j] + 1\n"
            string = string + "                else:\n"
            string = string + "                    votes[l+j+1] = votes[l+j+1] + 1\n"
            string = string + "                idx = idx + 1\n"
		
            # Prediction
            string = string + "        max_vote = 0\n"
            string = string + "        i_max_vote = 0\n"
            string = string + "        for k in range(len(votes)):\n"
            string = string + "            if votes[k]>max_vote:\n"
            string = string + "                max_vote, i_max_vote = votes[k], k\n"
            string = string + "        y_hat[i] = classes[i_max_vote]\n"
            string = string + "    return y_hat\n"
           
        return string


    """
    Helper function to recursively compute the body of a DecisionTreeClassifier
    """
    def _treebody2str(self, estimator, node_id, depth):

        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature
        threshold      = estimator.tree_.threshold
        
        my_string = ""
        
        if children_left[node_id] == children_right[node_id]:
            
            # It is a leaf
            my_string = my_string + '%sreturn %s\n' % (' '*depth*4, estimator.classes_[np.argmax(estimator.tree_.value[node_id][0])])

        else:

            # Print the decision to take at this level
            my_string = my_string + '%sif X%d < %.3f:\n' % (' '*depth*4, (feature[node_id]+1), threshold[node_id])
            my_string = my_string + self._treebody2str(estimator, children_left[node_id],  depth+1)
            my_string = my_string + '%selse:\n' % (' '*depth*4)
            my_string = my_string + self._treebody2str(estimator, children_right[node_id], depth+1)
                
        return my_string


    """
    Convert a DecisionTreeClassifier into a string
    """
    def _DecisionTreeClassifier(self, estimator):

        # TODO: sanity check over estimator
        
        n_nodes        = estimator.tree_.node_count
        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature

        tree_string = ""
        
        # Compute the tree header
        
        features_set = set()
                
        for node_id in range(n_nodes):

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                features_set.add('X%d' % (feature[node_id]+1))
        
        tree_string = tree_string + "def tree" + str(features_set) + ":\n"

        # Compute the tree body
        tree_string = tree_string + self._treebody2str(estimator, 0, 1)

        return tree_string


    """
    Convert a MLPClassifier into a string
    """
    def _MLPClassifier(self, estimator):
        
        # TODO: sanity check over estimator
        
        # TODO: Computation code should be optimized
        
        # TODO: Provide support to other activation functions
                
        #
        # Discretize coeficients
        #
        
        annw = []        
        for layer in estimator.coefs_:
            for node in layer:
                for coef in node:
                    annw.append(coef)

        annw = np.array(annw)
        annw = _discretize_vector(annw)
        
        ind  = 0
        coefs = list()
        for i in np.arange(len(estimator.coefs_)):
            layer = list()
            for j in np.arange(len(estimator.coefs_[i])):
                node = list()
                for k in np.arange(len(estimator.coefs_[i][j])):
                    node.append(annw[ind])
                    ind = ind + 1
                layer.append(node)
            coefs.append(layer)
            
        #
        # Discretize intercepts
        #
                    
        annb = []
        for layer in estimator.intercepts_:
            for node in layer:
                annb.append(node)

        annb = np.array(annb)
        annb = _discretize_vector(annb)
        
        ind  = 0
        inters = list()
        for i in np.arange(len(estimator.intercepts_)):
            layer = list()
            for j in np.arange(len(estimator.intercepts_[i])):
                layer.append(annb[ind])
                ind = ind + 1
            inters.append(layer)
                    
        #
        # Create the model
        #

        # Header
        string = "def NN(X):\n"
 
        # Weights
        string = string + "    W["
        for i in np.arange(len(coefs)):
            string = string + str(coefs[i]) + ", "
        string = string + "]\n"
            
        # Bias
        string = string + "    b["        
        for i in np.arange(len(coefs)):
            string = string + str(inters[i]) + ", "
        string = string + "]\n"
       
        # First layer
        
        string = string + "    Z = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(W[0].shape[0]):\n"
        string = string + "        for j in range(W[0].shape[1]):\n"
        string = string + "            Z[i] = Z[i] + W[0, i, j] * X[j]\n"
        string = string + "        Z[i] = Z[i] + b[0][i] \n"
            
        string = string + "    A = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(Z.shape[0]):\n"
        string = string + "        A[i] = max(Z[i], 0)\n"
        
        # Hiddent layers
        
        string = string + "    for i in range(1, " + str(len(estimator.coefs_)) + "):\n"
            
        string = string + "        Z = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(W[i].shape[0]):\n"
        string = string + "            for k in range(W[i].shape[1]):\n"
        string = string + "                Z[j] = Z[j] + W[i, j, k] * A[k]\n"
        string = string + "            Z[j] = Z[j] + b[i][j] \n"
            
        string = string + "        A = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(Z.shape[0]):\n"
        string = string + "            A = max(Z[j], 0)\n"
        
        # Predictions
        
        string = string + "    softmax = 0\n"
        string = string + "    prediction = 0\n"
        string = string + "    totalmax = 0\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        totalmax = totalmax + exp(A[i])\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        newmax = exp(A[i])\n"        
        string = string + "        if newmax > softmax:\n"        
        string = string + "            softmax = newmax \n"
        string = string + "            prediction = i\n"
        
        string = string + "    return prediction\n"

        return string
    

    """
    Convert a LinearRegression into a string
    """
    def _LinearRegression(self, estimator):

        #
        # Retrieve weigths
        #
                
        coefs     = estimator.coef_
        intercept = estimator.intercept_
        
        # Header
        string = "def LinearRegression(X):\n"
             
        # Similarities
        string = string + "    W = ["        
        for i in np.arange(len(coefs)):
            string = string + str(coefs[i]) + ", "
        string = string + "]\n"
        string = string + "    b = "
        string = string + str(intercept) + "\n"
            
        string = string + "    y_hat    = 0\n"
        string = string + "    for i in range(len(W)):\n"
        string = string + "        y_hat = W[i] * X[i]\n"
        string = string + "    y_hat = y_hat + b\n"        
        string = string + "    return y_hat\n"
                
        return string


    """
    Helper function to recursively compute the body of a DecisionTreeRegressor
    """
    def _treeregressorbody2str(self, estimator, node_id, depth):

        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature
        threshold      = estimator.tree_.threshold
        
        my_string = ""
        
        if children_left[node_id] == children_right[node_id]:
            
            # It is a leaf
            my_string = my_string + '%sreturn %s\n' % (' '*depth*4, np.argmax(estimator.tree_.value[node_id][0]))

        else:

            # Print the decision to take at this level
            my_string = my_string + '%sif X%d < %.3f:\n' % (' '*depth*4, (feature[node_id]+1), threshold[node_id])
            my_string = my_string + self._treeregressorbody2str(estimator, children_left[node_id],  depth+1)
            my_string = my_string + '%selse:\n' % (' '*depth*4)
            my_string = my_string + self._treeregressorbody2str(estimator, children_right[node_id], depth+1)
            
        return my_string


    """
    Convert a LinearSVR into a string
    """
    def _LinearSVR(self, estimator):
        
        # TODO: Adapt to LinearSVR

        #
        # Discretize similarities
        #
        
        M = estimator.coef_
        M = M.flatten()
        M = _discretize_vector(M)
        M = np.array(M)
        M = M.reshape(estimator.coef_.shape)
        
        #
        # Create the model
        #

        # Header
        string = "def LinearSVC(X):\n"
             
        # Similarities
        string = string + "    M["        
        for i in np.arange(len(M)):
            string = string + str(M[i]) + ", "
        string = string + "]\n"

        string = string + "    y_hat    = None\n"
        string = string + "    max_prob = 0\n"
        string = string + "    for i in range(len(estimator.classes_)):\n"
        string = string + "        prob = 1\n"
        string = string + "        for j in range(len(M[i])):\n"
        string = string + "            prob = prob * M[i][j]\n"
        string = string + "        prob = py[i] *  prob\n"
        string = string + "        if prob > max_prob:\n"
        string = string + "            y_hat = estimator.classes_[i]\n"
        string = string + "    return y_hat\n"

        return string

    """
    Convert a DecisionTreeRegressor into a string
    """
    def _DecisionTreeRegressor(self, estimator):
        
        # TODO: sanity check over estimator
        
        n_nodes        = estimator.tree_.node_count
        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature

        tree_string = ""
        
        #
        # Compute the tree header
        #
        
        features_set = set()
                
        for node_id in range(n_nodes):

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                features_set.add('X%d' % (feature[node_id]+1))
        
        tree_string = tree_string + "def DecisionTreeRegressor" + str(features_set) + ":\n"

        #
        # Compute the tree body
        # 
        
        tree_string = tree_string + self._treeregressorbody2str(estimator, 0, 1)

        return tree_string

        
    """
    Convert a MLPRegressor into a string
    """
    def _MLPRegressor(self, estimator):
        
        # TODO: Adapt to MLPRegressor
    
        # TODO: sanity check over estimator
        
        # TODO: Computation code should be optimized
        
        # TODO: Provide support to other activation functions
                
        #
        # Discretize coeficients
        #
        
        annw = []        
        for layer in estimator.coefs_:
            for node in layer:
                for coef in node:
                    annw.append(coef)

        annw = np.array(annw)
        annw = _discretize_vector(annw)
        
        ind  = 0
        coefs = list()
        for i in np.arange(len(estimator.coefs_)):
            layer = list()
            for j in np.arange(len(estimator.coefs_[i])):
                node = list()
                for k in np.arange(len(estimator.coefs_[i][j])):
                    node.append(annw[ind])
                    ind = ind + 1
                layer.append(node)
            coefs.append(layer)
            
        #
        # Discretize intercepts
        #
                    
        annb = []
        for layer in estimator.intercepts_:
            for node in layer:
                annb.append(node)

        annb = np.array(annb)
        annb = _discretize_vector(annb)
        
        ind  = 0
        inters = list()
        for i in np.arange(len(estimator.intercepts_)):
            layer = list()
            for j in np.arange(len(estimator.intercepts_[i])):
                layer.append(annb[ind])
                ind = ind + 1
            inters.append(layer)
                    
        #
        # Create the model
        #

        # Header
        string = "def NN(X):\n"
 
        # Weights
        string = string + "    W["
        for i in np.arange(len(coefs)):
            string = string + str(coefs[i]) + ", "
        string = string + "]\n"
            
        # Bias
        string = string + "    b["        
        for i in np.arange(len(coefs)):
            string = string + str(inters[i]) + ", "
        string = string + "]\n"
       
        # First layer
        
        string = string + "    Z = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(W[0].shape[0]):\n"
        string = string + "        for j in range(W[0].shape[1]):\n"
        string = string + "            Z[i] = Z[i] + W[0, i, j] * X[j]\n"
        string = string + "        Z[i] = Z[i] + b[0][i] \n"
            
        string = string + "    A = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(Z.shape[0]):\n"
        string = string + "        A[i] = max(Z[i], 0)\n"
        
        # Hiddent layers
        
        string = string + "    for i in range(1, " + str(len(estimator.coefs_)) + "):\n"
            
        string = string + "        Z = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(W[i].shape[0]):\n"
        string = string + "            for k in range(W[i].shape[1]):\n"
        string = string + "                Z[j] = Z[j] + W[i, j, k] * A[k]\n"
        string = string + "            Z[j] = Z[j] + b[i][j] \n"
            
        string = string + "        A = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(Z.shape[0]):\n"
        string = string + "            A = max(Z[j], 0)\n"
        
        # Predictions
        
        string = string + "    softmax = 0\n"
        string = string + "    prediction = 0\n"
        string = string + "    totalmax = 0\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        totalmax = totalmax + exp(A[i])\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        newmax = exp(A[i])\n"        
        string = string + "        if newmax > softmax:\n"        
        string = string + "            softmax = newmax \n"
        string = string + "            prediction = i\n"
        
        string = string + "    return prediction\n"

        return string        
    
#
# Class Nescience
# 
       
class Nescience(BaseEstimator):

    def __init__(self, X_type="numeric", y_type="numeric", compressor="bz2", method="Harmonic"):

        valid_X_types = ("numeric", "mixed", "categorical")
        valid_y_types = ("numeric", "categorical")

        if X_type not in valid_X_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_X_types, X_type))

        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        self.X_type     = X_type
        self.y_type     = y_type
        self.compressor = compressor
        self.method     = method

        return None

    
    def fit(self, X, y):
        """
        Initialization of the class nescience
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.

        method (string):     method used to comput the nescience. Valid
                             values are: "Euclid", "Arithmetic",
                             "Geometric", "Product", "Addition" and
                             "Harmonic".
                             
        compressor (string): compressor used to compute redudancy. Valid
                             values are: "bz2", "lzma" and "zlib".
          
        """
		
        X, y = check_X_y(X, y, dtype=None)

        self.miscoding_  = Miscoding(X_type=self.X_type, y_type=self.y_type, redundancy=False)
        self.miscoding_.fit(X, y)

        self.inaccuracy_ = Inaccuracy(y_type=self.y_type)
        self.inaccuracy_.fit(X, y)        

        self.surfeit_    = Surfeit(y_type=self.y_type, compressor=self.compressor)
        self.surfeit_.fit(X, y)
        
        return self


    def nescience(self, model, subset=None, predictions=None, model_string=None):
        """
        Compute the nescience of a model
        
        Parameters
        ----------
        model       : a trained model

        subset      : array-like, shape (n_features)
                      1 if the attribute is in use, 0 otherwise
                      If None, attributes will be infrerred throught model
                      
        model_str   : a string based representation of the model
                      If None, string will be derived from model
                    
        Returns
        -------
        Return the nescience (float)
        """
        
        check_is_fitted(self)

        if subset is None:
            miscoding = self.miscoding_.miscoding_model(model)
        else:
            miscoding = self.miscoding_.miscoding_subset(subset)

        if predictions is None:
            inaccuracy = self.inaccuracy_.inaccuracy_model(model)
        else:
            inaccuracy = self.inaccuracy_.inaccuracy_predictions(predictions)
            
        if model_string is None:
            surfeit = self.surfeit_.surfeit_model(model)
        else:
            surfeit = self.surfeit_.surfeit_string(model_string)            

        # Avoid dividing by zero
        
        if surfeit == 0:
            surfeit = 10e-6
    
        if inaccuracy == 0:
            inaccuracy = 10e-6

        if miscoding == 0:
            miscoding = 10e-6
            
        # TODO: Think about this problem
        if surfeit < inaccuracy:
            # The model is still too small to use surfeit
            surfeit = 1

        # Compute the nescience according to the method specified by the user
        if self.method == "Euclid":
            # Euclidean distance
            nescience = math.sqrt(miscoding**2 + inaccuracy**2 + surfeit**2)
        elif self.method == "Arithmetic":
            # Arithmetic mean
            nescience = (miscoding + inaccuracy + surfeit) / 3
        elif self.method == "Geometric":
            # Geometric mean
            nescience = math.pow(miscoding * inaccuracy * surfeit, 1/3)
        elif self.method == "Product":
            # The product of both quantities
            nescience = miscoding * inaccuracy * surfeit
        elif self.method == "Addition":
            # The product of both quantities
            nescience = miscoding + inaccuracy + surfeit
        # elif self.method == "Weighted":
            # Weigthed sum
            # TODO: Not yet supported
            # nescience = self.weight_miscoding * miscoding + self.weight_inaccuracy * inaccuracy + self.weight_surfeit * surfeit
        elif self.method == "Harmonic":
            # Harmonic mean
            nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/surfeit))
        # else -> rise exception
        
        return nescience


class AutoClassifier(BaseEstimator, ClassifierMixin):
    
    # TODO: Class documentation
    
    def __init__(self, auto=True, random_state=None):
        
        self.random_state = random_state
        self.auto = auto
        
        return None

    
    def fit(self, X, y):
        """
        Select the best model that explains y given X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values as numbers.
        
        auto: find automatically the optimal model
            
        Returns
        -------
        self
        """        
                
        # Supported Classifiers
        
        self.classifiers_ = [
            self.MultinomialNB,
            self.DecisionTreeClassifier,
            self.LinearSVC,
            self.SVC,            
            self.MLPClassifier
        ]

        self.X_, self.y_ = check_X_y(X, y, dtype=None)
        # check_classification_targets(self.y_)

        self.nescience_ = Nescience(X_type="numeric", y_type="categorical")
        self.nescience_.fit(self.X_, self.y_)
        
        # new y contains class indexes rather than labels in the range [0, n_classes]
        self.classes_, self.y_ = np.unique(self.y_, return_inverse=True)						  
        
        nsc = 1
        self.model_ = None
        self.viu_   = None
        
        # Find optimal model
        if self.auto:
        
            for clf in self.classifiers_:
            
                # TODO: print classifier if verbose
                print("Classifier: " + str(clf), end='')
                
                # If X contains negative values, MultinomialNB is skipped
                if clf == self.MultinomialNB and not (self.X_>=0).all():
                    # TODO: Should be based on a verbose flag
                    print("Skipped!")                
                    continue
                
                (new_nsc, new_model, new_viu) = clf()

                # TODO: Should be based on a verbose flag
                print("Nescience:", new_nsc)                

                if new_nsc < nsc:
                    nsc   = new_nsc
                    self.model_ = new_model
                    self.viu_   = new_viu
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return a list of classes predicted
        """
        
        check_is_fitted(self)
        X = check_array(X)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]

        return self.classes_[self.model_.predict(msdX)]


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
      
        Return an array of probabilities. The order of the list match the order
        the internal attribute classes_
        """
        
        check_is_fitted(self)
        X = check_array(X)

        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]
					
        return self.model_.predict_proba(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        check_is_fitted(self)
        X, y = check_X_y(X, y, dtype=None)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]        
        
        return self.model_.score(msdX, y)
	
	
    def get_model(self):
        """
        Get access to the private attribute model
		
        Return self.model_
        """
        return self.model_


    def MultinomialNB(self):
        
        # No hyperparameters to optimize
        
        model = MultinomialNB()
        model.fit(self.X_, self.y_)

        nsc = self.nescience_.nescience(model)
            
        return (nsc, model, None)

    
    def LinearSVC(self):
        
        # No hyperparameters to optimize
        
        model = SVC(kernel='linear', probability=True, random_state=self.random_state, decision_function_shape='ovo')
        model.fit(self.X_, self.y_)

        nsc = self.nescience_.nescience(model)
            
        return (nsc, model, None)


    def SVC(self): 
       
        # Different searches are possible to find the best hyperparameters
        # The following one produced good results on the three datasets it was tested on
        
        # The four hyperparameters to optimize
        hyper_param_default = ['degree', 'C', 'gamma', 'coef0']
		
        # The order in which they will be treated in the search
        hyper_param_order = [1, 2, 3, 4]
		
        # gamma is searched among the same values as C and coef0 but is always multiplied by inv (its default value)
        inv = 1/(len(self.X_[0])*np.var(self.X_))
		
        # maximum number of iterations to fit the SVC models in this search. Could be reduced to 1e5 or 1e4
        max_iter = 1e6

        hyper_param = [] 
        for i in range(4):
            hyper_param.append(hyper_param_default[hyper_param_order[i]-1])
        
        # Default values
        param_value = {'degree': 5, 'C': 1, 'gamma': inv, 'coef0': 1}
		
        tmp_model = SVC(kernel='poly', max_iter=max_iter)
        tmp_model.set_params(**param_value)
        tmp_model.fit(self.X_, self.y_)
        tmp_nsc = self.nescience_.nescience(tmp_model)
    
        decreased = True
        while (decreased):
        
            decreased = False
            
            for param in hyper_param:
                
                if param=='degree':
                
                    # Test degree=degree+1
                    tmp_model.set_params(**{param: param_value[param]+1})
                    tmp_model.fit(self.X_, self.y_)
                    new_nsc = self.nescience_.nescience(tmp_model)
                    if new_nsc<tmp_nsc:
                        tmp_nsc = new_nsc
                        param_value[param] += 1
                        decreased = True
                    else:
					
                        # Test degree=degree-1
                        tmp_model.set_params(**{param: param_value[param]-1})
                        tmp_model.fit(self.X_, self.y_)
                        new_nsc = self.nescience_.nescience(tmp_model)
                        if new_nsc<tmp_nsc:
                            tmp_nsc = new_nsc
                            decreased = True
                            param_value[param] -= 1
                        else:
				
                            # Test degree=degree+2
                            tmp_model.set_params(**{param: param_value[param]+2})
                            tmp_model.fit(self.X_, self.y_)
                            new_nsc = self.nescience_.nescience(tmp_model)
                            if new_nsc<tmp_nsc:
                                tmp_nsc = new_nsc
                                param_value[param] += 2
                                decreased = True
                            else:
					
                                # Test degree=degree-2
                                tmp_model.set_params(**{param: param_value[param]-2})
                                tmp_model.fit(self.X_, self.y_)
                                new_nsc = self.nescience_.nescience(tmp_model)
                                if new_nsc<tmp_nsc:
                                    tmp_nsc = new_nsc
                                    decreased = True
                                    param_value[param] -= 2
                                else:
                                    tmp_model.set_params(**{param: param_value[param]})
                
                else: # param = 'C' or 'coef0' or 'gamma'
                
                    # Test param=param*2
                    tmp_model.set_params(**{param: param_value[param]*2})
                    tmp_model.fit(self.X_, self.y_)
                    new_nsc = self.nescience_.nescience(tmp_model)
                    if new_nsc<tmp_nsc:
                        tmp_nsc = new_nsc
                        param_value[param] *= 2
                        decreased = True
                    else:
					
                        # Test param=param/2
                        tmp_model.set_params(**{param: param_value[param]/2})
                        tmp_model.fit(self.X_, self.y_)
                        new_nsc = self.nescience_.nescience(tmp_model)
                        if new_nsc<tmp_nsc:
                            tmp_nsc = new_nsc
                            param_value[param] /= 2
                            decreased = True
                        else:
                            tmp_model.set_params(**{param: param_value[param]})
                            if param=='coef0':
							
                                # Test coef0=-coef0
                                tmp_model.set_params(**{param: -param_value[param]})
                                tmp_model.fit(self.X_, self.y_)
                                new_nsc = self.nescience_.nescience(tmp_model)
                                if new_nsc<tmp_nsc:
                                    tmp_nsc = new_nsc
                                    param_value[param] *= -1
                                    decreased = True
                                else:
                                    tmp_model.set_params(**{param: param_value[param]})
    

        model = tmp_model.fit(self.X_, self.y_)
        nsc = tmp_nsc
        
        if self.auto==False:
            self.model_ = model
            
        return (nsc, model, None)


    def DecisionTreeClassifier(self):

        clf  = DecisionTreeClassifier()
        path = clf.cost_complexity_pruning_path(self.X_, self.y_)

        previous_nodes = -1
        best_nsc       = 1
        best_model     = None

        # For every possible prunning point in reverse order
        for ccp_alpha in reversed(path.ccp_alphas):
    
            model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=self.random_state)
            model.fit(self.X_, self.y_)
    
            # Skip if nothing has changed
            if model.tree_.node_count == previous_nodes:
                continue
    
            previous_nodes = model.tree_.node_count
    
            new_nsc = self.nescience_.nescience(model)
    
            if new_nsc < best_nsc:
                best_nsc   = new_nsc
                best_model = model
            else:
                break
        return (best_nsc, best_model, None)

    
    def MLPClassifier(self):
        
        # Relevance of features
        tmp_msd = msd = self.nescience_.miscoding_.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Create the initial neural network
        #  - two features
        #  - one hidden layer
        #  - three units
        
        tmp_hu = hu = [3]

        # Select the two most relevant features
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1
        viu[np.argmax(msd)] = 1
        msd[np.where(viu)] = -1
        
        msdX = self.X_[:,np.where(viu)[0]]
        tmp_nn = nn = MLPClassifier(hidden_layer_sizes = hu, random_state=self.random_state)
        nn.fit(msdX, self.y_)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self.nescience_.nescience(nn, subset=viu, predictions=prd)
        
        # While the nescience decreases
        decreased = True        
        while (decreased):
            
            decreased = False

            #
            # Test adding a new feature  
            #
            
            # Check if therer are still more variables to add
            if np.sum(viu) != viu.shape[0]:
            
                new_msd = msd.copy()
                new_viu = viu.copy()
            
                new_viu[np.argmax(new_msd)] = 1
                new_msd[np.where(viu)] = -1

                msdX    = self.X_[:,np.where(new_viu)[0]]
                new_nn  = MLPClassifier(hidden_layer_sizes = hu, random_state=self.random_state)        
                new_nn.fit(msdX, self.y_)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience_.nescience(new_nn, subset=new_viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = new_msd
                    tmp_viu = new_viu
                    tmp_hu  = hu
                    
            #
            # Test adding a new layer
            #
            
            new_hu = hu.copy()
            new_hu.append(3)

            msdX    = self.X_[:,np.where(viu)[0]]
            new_nn  = MLPClassifier(hidden_layer_sizes = new_hu, random_state=self.random_state)
            new_nn.fit(msdX, self.y_)
            prd     = new_nn.predict(msdX)
            new_nsc = self.nescience_.nescience(new_nn, subset=viu, predictions=prd)
            
            # Save data if nescience has been reduced 
            if new_nsc < tmp_nsc:                                
                decreased = True
                tmp_nn  = new_nn
                tmp_nsc = new_nsc
                tmp_msd = msd
                tmp_viu = viu
                tmp_hu  = new_hu

            #
            # Test adding a new unit
            #
            
            for i in np.arange(len(hu)):
                
                new_hu    = hu.copy()
                new_hu[i] = new_hu[i] + 1            

                msdX    = self.X_[:,np.where(viu)[0]]
                new_nn  = MLPClassifier(hidden_layer_sizes = new_hu, random_state=self.random_state)
                new_nn.fit(msdX, self.y_)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience_.nescience(new_nn, subset=viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = msd
                    tmp_viu = viu
                    tmp_hu  = new_hu
                
            # Update neural network
            nn      = tmp_nn
            nsc     = tmp_nsc
            viu     = tmp_viu
            msd     = tmp_msd
            hu      = tmp_hu

        # -> end while

        return (nsc, nn, viu)


class AutoRegressor(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self, auto=True, random_state=None):
        
        self.random_state = random_state
        self.auto = auto
        
        return None

    
    def fit(self, X, y):
        """
        Select the best model that explains y given X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.
        
        auto: find automatically the optimal model
            
        Returns
        -------
        self
        """
        
        # Supported Regressors
        
        self.regressors_ = [
            self.LinearRegression,
            self.LinearSVR,
            self.DecisionTreeRegressor,
            self.MLPRegressor
        ]

        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        self.nescience_ = Nescience(X_type="numeric", y_type="numeric")
        self.nescience_.fit(self.X_, self.y_)
        
        nsc = 1
        self.model_ = None
        self.viu_   = None
        
        # Find automatically the optimal model
        
        if self.auto:
            
            for reg in self.regressors_:
            
                # TODO: Should be based on a verbose flag
                print("Regressor: " + str(reg), end='')
            
                (new_nsc, new_model, new_viu) = reg()
                
                print("Nescience:", new_nsc)
            
                if new_nsc < nsc:
                    nsc   = new_nsc
                    self.model_ = new_model
                    self.viu_   = new_viu
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return the predicted value
        """
        
        check_is_fitted(self)
        X = check_array(X)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]
                
        return self.model_.predict(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """

        check_is_fitted(self)
        X = check_array(X)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]        
        
        return self.model_.score(msdX, y)


    def get_model(self):
        """
        Get access to the private attribute model
		
        Return self.model_
        """
        return self.model_
		
		
    def LinearRegression(self):
        
        # Relevance of features
        msd = self.nescience_.miscoding_.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self.y_)
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
            
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self.y_)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced                        
            if new_nsc < nsc:                                
                decreased = True
                model     = new_model
                nsc       = new_nsc
                msd       = new_msd
                viu       = new_viu
        
        return (nsc, model, viu)


    def LinearSVR(self):
        # TODO: Optimize hyperparameters
                    
        # model = LinearSVR(multi_class="crammer_singer")
        model = LinearSVR(random_state=self.random_state)
        model.fit(self.X_, self.y_)

        nsc = self.nescience_.nescience(model)
            
        return (nsc, model, None)    


    def DecisionTreeRegressor(self):
        
        clf  = DecisionTreeRegressor(random_state=self.random_state)
        path = clf.cost_complexity_pruning_path(self.X_, self.y_)

        previous_nodes = -1
        best_nsc       = 1
        best_model     = None
        
        # For every possible prunning point in reverse order
        for ccp_alpha in reversed(path.ccp_alphas):
                
            model = DecisionTreeRegressor(ccp_alpha=ccp_alpha, random_state=self.random_state)
            model.fit(self.X_, self.y_)
    
            # Skip if nothing has changed
            if model.tree_.node_count == previous_nodes:
                continue
    
            previous_nodes = model.tree_.node_count
    
            new_nsc = self.nescience_.nescience(model)
            
            if new_nsc < best_nsc:
                best_nsc   = new_nsc
                best_model = model
            else:
                break
    
        return (best_nsc, best_model, None)       


    def MLPRegressor(self):
        
        # Relevance of features
        tmp_msd = msd = self.nescience_.miscoding_.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Create the initial neural network
        #  - two features
        #  - one hidden layer
        #  - three units
        
        tmp_hu = hu = [3]

        # Select the two most relevant features
        viu[np.argmax(msd)] =  1        
        msd[np.where(viu)]  = -1
        viu[np.argmax(msd)] =  1
        msd[np.where(viu)]  = -1
        
        msdX = self.X_[:,np.where(viu)[0]]
        tmp_nn = nn = MLPRegressor(hidden_layer_sizes = hu, random_state=self.random_state)
        nn.fit(msdX, self.y_)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self.nescience_.nescience(nn, subset=viu, predictions=prd)
        
        # While the nescience decreases
        decreased = True        
        while (decreased):
                        
            decreased = False

            #
            # Test adding a new feature  
            #
            
            # Check if therer are still more variables to add
            if np.sum(viu) != viu.shape[0]:
            
                new_msd = msd.copy()
                new_viu = viu.copy()
            
                new_viu[np.argmax(new_msd)] = 1
                new_msd[np.where(viu)] = -1

                msdX    = self.X_[:,np.where(new_viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = hu, random_state=self.random_state)        
                new_nn.fit(msdX, self.y_)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience_.nescience(new_nn, subset=new_viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = new_msd
                    tmp_viu = new_viu
                    tmp_hu  = hu
                    
            #
            # Test adding a new layer
            #
            
            new_hu = hu.copy()
            new_hu.append(3)

            msdX    = self.X_[:,np.where(viu)[0]]
            new_nn  = MLPRegressor(hidden_layer_sizes = new_hu, random_state=self.random_state)
            new_nn.fit(msdX, self.y_)
            prd     = new_nn.predict(msdX)
            new_nsc = self.nescience_.nescience(new_nn, subset=viu, predictions=prd)
            
            # Save data if nescience has been reduced 
            if new_nsc < tmp_nsc:                                
                decreased = True
                tmp_nn  = new_nn
                tmp_nsc = new_nsc
                tmp_msd = msd
                tmp_viu = viu
                tmp_hu  = new_hu

            #
            # Test adding a new unit
            #
            
            for i in np.arange(len(hu)):
                
                new_hu    = hu.copy()
                new_hu[i] = new_hu[i] + 1            

                msdX    = self.X_[:,np.where(viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = new_hu, random_state=self.random_state)
                new_nn.fit(msdX, self.y_)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience_.nescience(new_nn, subset=viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = msd
                    tmp_viu = viu
                    tmp_hu  = new_hu
                
            # Update neural network
            nn      = tmp_nn
            nsc     = tmp_nsc
            viu     = tmp_viu
            msd     = tmp_msd
            hu      = tmp_hu

        # -> end while

        return (nsc, nn, viu)

    # WARNING: Experimental, do not use in production
    # TODO: build a sklearn wrapper around the model
    def GrammaticalEvolution(self):
        
        # A grammar is a dictionary keyed by non terminal symbols
        #     Each value is a list with the posible replacements
        #         Each replacement contains a list with tokens
        #
        # The grammar in use is:
        #
        #     <expression> ::= self.X_[:,<feature>] |
        #                      <number> <scale> self.X_[:,<feature>] |
        #                      self.X_[:,<feature>]) ** <exponent> |
        #                      (<expression>) <operator> (<expression>)
        #                 
        #     <operator>   ::= + | - | * | /
        #     <scale>      ::= *
        #     <number>     ::= <digit> | <digit><digit0> | | <digit><digit0><digit0>
        #     <digit>      ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        #     <digit0>     ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        #     <exponent>   ::= 2 | 3 | (1/2) | (1/3)
        #     <feature>    ::= 1 .. self.X_.shape[1]

        self.grammar = {
            "expression": [
                            ["self.X_[:,", "<feature>", "]"],
                            ["<number>", "<scale>", "self.X_[:,", "<feature>", "]"],
                            ["self.X_[:,", "<feature>", "]**", "<exponent>"],
                            ["(", "<expression>", ")", "<operator>", "(", "<expression>", ")"]
                          ],
            "operator":   ["+", "-", "*", "/"],
            "scale":      ["*"],
            "number":     [
                            ["<digit>"], 
                            ["<digit>", "<digit0>"],
                            ["<digit>", "<digit0>", "<digit0>"]
                          ],
            "digit":      ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "digit0":     ["0", "5"],
            "exponent":   ["2", "3", "(1/2)", "(1/3)"],
            "feature":    None
        }

        # Fill in features         
        self.grammar["feature"] = [str(i) for i in np.arange(0, self.X_.shape[1])]

        self.max_num_tokens  = 10 # Sufficient to cover all possible tokens from grammar
        self.max_num_derivations = self.max_num_tokens * self.max_num_tokens # TODO: Think about that

        # Use differential evolution to find the optimal model
        bounds = [(0, self.max_num_tokens)] * self.max_num_derivations
        result = differential_evolution(self._evaluate_genotype, bounds)
        
        # Retrieve model
        model = self._parse_grammar(result.x)
        
        # Compute the predicted values
        pred = eval(model)

        # Compute model string
        model_str = model.replace("self.", "")
        
        # Compute the variables in use
        viu          = np.zeros(self.X_.shape[1], dtype=int)                    
        match        = re.compile(r'self.X_\[:,(\d+)\]') 
        indices      = match.findall(model) 
        indices      = [int(i) for i in indices] 
        viu[indices] = 1

        # Compute the nescience
        nsc = self.nescience_.nescience(None, subset=viu, predictions=pred, model_string=model_str)
        
        return (nsc, model, viu)


    """
    Given a genotype (a list of integers) compute the nescience of the
    corresponding phenotype given the grammar.
    
    Return the nescience of the phenotype
    """
    def _evaluate_genotype(self, x):
                
        # Retrieve model
        model = self._parse_grammar(x)
                
        # Compute the predicted values
        try:
            pred = eval(model)
        except:
            # In case of non-evaluable model, return a nescience of 1
            return 1 
                            
        # Compute a simplified version of model string
        model_str = model.replace("self.", "")
                
        # Compute the variables in use
        viu          = np.zeros(self.X_.shape[1], dtype=int)                    
        match        = re.compile(r'self.X_\[:,(\d+)\]') 
        indices      = match.findall(model) 
        indices      = [int(i) for i in indices] 
        viu[indices] = 1
        
        # Compute the nescience
        try:
            nsc = self.nescience_.nescience(None, subset=viu, predictions=pred, model_string=model_str)
        except:
            # In case of non-computable nesciencee, return a value of 1
            return 1 
                
        return nsc


    """
    Given a genotype (a list of integers) compute the  corresponding phenotype
    given the grammar.
    
    Return a string based phenotype
    """
    def _parse_grammar(self, x):
        
        x = [int(round(i)) for i in x]
        
        phenotype = ["<expression>"]
        ind       = 0
        modified  = True
        
        # Meanwhile there are no more non-terminal symbols
        while modified:
            
            modified = False
            new_phenotype = list()
                        
            for token in phenotype:
                            
                if token[0] == '<' and token[-1] == '>':
                    
                    token     = token[1:-1]
                    new_token = self.grammar[token][x[ind] % len(self.grammar[token])]
                                        
                    if type(new_token) == str:
                        new_token = list(new_token)
                                            
                    new_phenotype = new_phenotype + new_token
                    modified = True
                    ind = ind + 1
                    ind = ind % self.max_num_derivations
                                        
                else:
                                   
                    # new_phenotype = new_phenotype + list(token)
                    new_phenotype.append(token)
                         
            phenotype = new_phenotype
                    
        model = "".join(phenotype)

        return model

        
class AutoTimeSeries(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self, auto=True):
        
        self.auto = auto
		
        return None

    
    # TODO: provide support to autofit
    def fit(self, ts):
        """
        Select the best model that explains the time series ts.
        
        Parameters
        ----------            
        ts : array-like, shape (n_samples)
            The time series as numbers.
        auto: compute automatically the optimal model
            
        Returns
        -------
        self
        """

        # Supported time series models
        
        self.models_ = [
            self.AutoRegressive,
            self.MovingAverage,
            self.ExponentialSmoothing
        ]

        self.X_, self.y_ = self._whereIsTheX(ts)

        self.nescience_ = Nescience(X_type="numeric", y_type="numeric")
        self.nescience_.fit(self.X_, self.y_)
        
        nsc = 1
        self.model_ = None
        self.viu_   = None

        # Find optimal model
        if self.auto:
        
            for reg in self.models_:
            
                (new_nsc, new_model, new_viu) = reg()
            
                if new_nsc < nsc: 
                    nsc   = new_nsc
                    self.model_ = new_model
                    self.viu_   = new_viu
        
        return self


    """
       Transfrom a unidimensional time series ts into a classical X, y dataset
       
       * size: size of the X, that is, number of attributes
    """
    def _whereIsTheX(self, ts, size=None):
                
        X = list()
        y = list()

        lts = len(ts)
        
        if size == None:
            size = int(np.sqrt(lts))

        for i in np.arange(lts - size):
            X.append(ts[i:i+size])
            y.append(ts[i+size])
            
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    

    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return the predicted value
        """
        
        check_is_fitted(self)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]
                
        return self.model_.predict(msdX)


    def score(self, ts):
        """
        Evaluate the performance of the current model given a test time series

        Parameters
        ----------            
        ts : array-like, shape (n_samples)
            The time series as numbers.
            
        Returns
        -------    
        Return one minus the mean error
        """
        
        check_is_fitted(self)

        X, y = self._whereIsTheX(ts)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]        
        
        return self.model_.score(msdX, y)
		
		
    def get_model(self):
        """
        Get access to the private attribute model
		
        Return self.model_
        """
        return self.model_

		
    def AutoRegressive(self):
        
        # Relevance of features
        msd = self.nescience_.miscoding_.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model        
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self.y_)
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
                        
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self.y_)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced                        
            if new_nsc < nsc:                                
                decreased = True
                model     = new_model
                nsc       = new_nsc
                msd       = new_msd
                viu       = new_viu
        
        return (nsc, model, viu)


    def MovingAverage(self):
        
        # Variables in use
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self.X_.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([1/i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)


    def ExponentialSmoothing(self):
        
        alpha = 0.2
        
        # Variables in use
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self.X_.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([(1-alpha)**i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)


class IncompressibleClassifier():
    
    # TODO: Class documentation
    
    def __init__(self, auto=False, random_state=None):

        # TODO: Document
        
        self.auto         = auto
        self.random_state = random_state

        return None


    def fit(self, X, y, model=None):

        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        # TODO: check it is a valid model
        # TODO: train if the model if auto = True

        self.model = model

        y_hat = self.model.predict(self.X_)
        self.incompressible = np.where(self.y_ != y_hat)[0]

        return


    def fit_classification(self):

        if self.y_isnumeric:

            regressor = AutoRegressor()

        else:

            model = AutoClassifier()

        return


    def get_incompressible(self):

        # TODO: Document

        return self.incompressible


    def clusters(self, n_clusters="Auto", filter_inertia=True, filter_repeated_attrs=True, filter_balancedness=True, filter_miscoding=True):

        # TODO: Check that the class is fitted
        # TODO: Allow to change the dimension of the cluster

        nis = len(self.incompressible)

        # Automatically select the number of clusters
        if n_clusters == "Auto":
            n_clusters = int(np.log2(nis)/2)

        # TODO : MinMaxScaler() ??
        km_model = KMeans(n_clusters=n_clusters)

        # Compute all possible clusters

        df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

        for i in np.arange(self.X_.shape[1]-1):
    
            for j in np.arange(i+1, self.X_.shape[1]):
        
                new_X = self.X_[np.ix_(self.incompressible,[i, j])]
                # new_X = scaler.fit_transform(new_X)
        
                km_model.fit(new_X)
        
                tmp_df = pd.DataFrame([{"Attribute 1": i, "Attribute 2": j, "Cluster": km_model, "Inertia": km_model.inertia_}])
                df = df.append(tmp_df, ignore_index=True)

        # TODO: Implement a filter based on Inertia

        # Filter repeated attributes

        if filter_repeated_attrs:

            attr_in_use = list()
            filtered_df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

            for index, row in df.sort_values(by=['Inertia']).iterrows():
    
                if (row["Attribute 1"] in attr_in_use) or (row["Attribute 2"] in attr_in_use):
                    continue
        
                attr_in_use.append(row["Attribute 1"])
                attr_in_use.append(row["Attribute 2"])
    
                tmp_df      = pd.DataFrame([{"Attribute 1": row["Attribute 1"], "Attribute 2": row["Attribute 2"], "Cluster": row["Cluster"], "Inertia": row["Inertia"]}])
                filtered_df = filtered_df.append(tmp_df, ignore_index=True)

            df = filtered_df

        # Filter non-balanced clusters

        if filter_balancedness:

            filter_ratio_low  = 0.2
            filter_ratio_high = 0.8

            filtered_df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

            for index, row in df.iterrows():

                new_X = self.X_[np.ix_(self.incompressible,[row["Attribute 1"], row["Attribute 2"]])]
                
                km_model = row["Cluster"]
                y_pred = km_model.predict(new_X)

                n_class_0 = np.sum(y_pred == 0)
                n_class_1 = np.sum(y_pred == 1)

                ratio = n_class_0 / (n_class_0 + n_class_1)

                if ratio < filter_ratio_low:
                    continue

                if ratio > filter_ratio_high:
                    continue

                tmp_df      = pd.DataFrame([{"Attribute 1": row["Attribute 1"], "Attribute 2": row["Attribute 2"], "Cluster": row["Cluster"], "Inertia": row["Inertia"]}])
                filtered_df = filtered_df.append(tmp_df, ignore_index=True)

            df = filtered_df

        # Filter attributes highly related

        if filter_miscoding:

            filtered_df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

            filter_miscoding  = 0.2

            mscd = Miscoding()
            mscd.fit(self.X_, self.y_)
            matrix = mscd.features_matrix()

            for index, row in df.iterrows():

                if matrix[row["Attribute 1"], row["Attribute 2"]] > filter_miscoding:
                    continue

                tmp_df      = pd.DataFrame([{"Attribute 1": row["Attribute 1"], "Attribute 2": row["Attribute 2"], "Cluster": row["Cluster"], "Inertia": row["Inertia"]}])
                filtered_df = filtered_df.append(tmp_df, ignore_index=True)

            df = filtered_df

        return df


