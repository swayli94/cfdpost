'''
Extract flow features of airfoils or wing sections.

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition

'''
import numpy as np
from sklearn.decomposition import PCA, KernelPCA


class PCASec():
    '''
    Extract features of airfoil wall Mach number distribution

    >>> pca = PCASec(n_point: int, n_feature=None, whiten=False, svd_solver='auto')

    PCA: principal component analysis

    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    ### Input:
    ```text
        n_point:        data size of an airfoil
        n_feature:      intensional size of features (or None)
        whiten:         whitening of the PCA
        svd_solver:     'auto', 'full', 'arpack', 'randomized'
    ```

    ### Attributes:
    ```text
        _low, _upp:     lower and upper bound of the data
        n_sample:       number of training samples
        x_train:        training samples, ndarray [n_sample, n_point]
        model:          estimator
    ```

    '''

    def __init__(self, n_point: int, n_feature=None, whiten=False, svd_solver='auto'):

        self.n_point = n_point
        self.n_feature = n_feature

        self._low = 0.0
        self._upp = 1.0

        self.model = PCA(n_components=n_feature, copy=True, whiten=whiten, svd_solver=svd_solver)

    def train(self, x_data):
        '''
        Fit the model with x_data

        ### Inputs:
        ```text
        x_data:     ndarray [n_sample, n_point]
        ```
        '''
        if not isinstance(x_data, np.ndarray):
            raise Exception('Must provide ndarray')
        elif x_data.shape[1] != self.n_point:
            raise Exception('Must provide ndarray, shape [n_sample, n_point]')
        
        self.n_sample = x_data.shape[0]
        self.x_train = x_data.copy()
        self.x_train = (self.x_train-self._low)/(self._upp-self._low)

        self.model.fit(self.x_train)
    
    def save_params(self, fname='MwSec_PCA.pth'):
        '''
        Save the parameters of this estimator
        '''
        params = self.model.get_params(deep=True) #type: dict

        with open(fname, 'w') as f:
            for k,v in sorted(params.items()):
                f.write(str(k)+' '+str(v)+'\n')

    def load_params(self, fname='MwSec_PCA.pth'):
        '''
        Load the parameters of this estimator
        '''
        params = {}

        with open(fname, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                k = line.split(' ')[0]
                v = line.split(' ')[1]
                params[k] = v

        self.model.set_params(params)

    def transform(self, x_data) -> np.ndarray:
        '''
        Apply dimensionality reduction to x_data

        ### Inputs:
        ```text
        x_data:     ndarray [n, n_point]
        ```
        ### Return:
        ```text
        data_:      ndarray [n, n_feature]
        ```
        '''
        if not isinstance(x_data, np.ndarray):
            raise Exception('Must provide ndarray')
        elif x_data.shape[1] != self.n_point:
            raise Exception('Must provide ndarray, shape [n, n_point]')

        data_ = (x_data-self._low)/(self._upp-self._low)

        return self.model.transform(data_)

    def inverse_transform(self, data_) -> np.ndarray:
        '''
        Transform data back to its original space

        ### Inputs:
        ```text
        data_:      ndarray [n, n_feature]
        ```
        ### Return:
        ```text
        x_data:     ndarray [n, n_point]
        ```
        '''
        if not isinstance(data_, np.ndarray):
            raise Exception('Must provide ndarray')
        elif data_.shape[1] != self.n_feature:
            raise Exception('Must provide ndarray, shape [n, n_feature]')

        data_ = self.model.inverse_transform(data_)
        data_ = data_*(self._upp-self._low) + self._low
    
        return data_


class KPCASec(PCASec):
    '''
    Extract features of airfoil wall Mach number distribution

    KPCA: kernel principal component analysis

    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

    ### Input:
    ```text
        n_point:        data size of an airfoil
        n_feature:      intensional size of features
        kernel:         'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'
        eigen_solver:   'auto', 'dense', 'arpack'   
        n_jobs:         number of parallel jobs to run (-1 means using all processors)
    ```

    ### Attributes:
    ```text
        _low, _upp:     lower and upper bound of the data
        n_sample:       number of training samples
        x_train:        training samples, ndarray [n_sample, n_point]
        model:          estimator
    ```

    '''

    def __init__(self, n_point: int, n_feature: int, kernel='linear', eigen_solver='auto', n_jobs=1):
        super().__init__(n_point, n_feature)

        self.model = KernelPCA(n_components=n_feature, kernel=kernel, eigen_solver=eigen_solver, copy_X=True, n_jobs=n_jobs)
    
    def save_params(self, fname='MwSec_KPCA.pth'):
        '''
        Save the parameters of this estimator
        '''
        super().save_params(fname=fname)

    def load_params(self, fname='MwSec_KPCA.pth'):
        '''
        Load the parameters of this estimator
        '''
        super().load_params(fname=fname)

