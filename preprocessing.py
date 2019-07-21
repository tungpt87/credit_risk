


class Normalizer:
    __cache = {}

    def __init__(self, method='min-max', path = None):
        if path is None:
            self.__cache['method'] = method
        else:
            load(path)

    '''
    Public functions
    '''
    def fit_transform(self, data, columns):
        if self.__cache['method'] == 'min-max':
            res, self.__cache['min'], self.__cache['max'] = self.__minMaxScaler(data, columns)

        for col in data.columns:
            if col not in columns:
                res[col] = data[col]

        return res

    def transform(self, data, columns):
        if self.__cache['method'] == 'min-max':
            res, _, _ = self.__minMaxScaler(data, columns, self.__cache['min'], self.__cache['max'])

        for col in data.columns:
            if col not in columns:
                res[col] = data[col]

        return res


    def load(self, path):
        self.__cache = np.load(path)[()]

    def save(self, path):
        import numpy as np
        np.save(path, self.__cache)

    '''
    Private functions
    ''' 

    def __minMaxScaler(self, data, columns, minVals = None, maxVals = None):
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import normalize

        if (minVals is None) and (maxVals is None):
            maxVals = np.max(data[columns], axis = 0) 
            minVals = np.min(data[columns], axis = 0)

        res = (data[columns] - minVals) / (maxVals - minVals)
        
        return res, minVals, maxVals


    
class Imputor:


    @classmethod
    def knnRegressor(cls, data, target, predictors, k=3, weights = 'distance'):
        parameters = {'k':k,
                      'weights': weights}

        return cls.__imputeMissingValues(data, target, predictors, 'knn-reg', parameters)

    @classmethod
    def knnClassifier(cls, data, target, predictors, k=3, weights = 'distance'):
        parameters = {'k':k,
                      'weights': weights}

        return cls.__imputeMissingValues(data, target, predictors, 'knn-cls', parameters)

    @classmethod 
    def __imputeMissingValues(cls, X, target, predictors = None, method = 'knn-reg', parameters = {'k':3}):
        import numpy as np
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        import pandas as pd
        
        X_wo_NAs = X[~X[target].isna()]
        X_w_NAs = X[X[target].isna()]
        
        if predictors is None:
            predictors = [col for col in W_wo_NAs if col != target]
        
        algo = None
        
        if method == 'knn-reg':
            algo = KNeighborsRegressor(parameters['k'], weights = parameters['weights'])
        if method == 'knn-cls':
            algo = KNeighborsClassifier(parameters['k'], weights = parameters['weights'])
        else:
            pass
        
        model = algo.fit(X_wo_NAs[predictors], X_wo_NAs[target])
        
        X_w_NAs[target] = model.predict(X_w_NAs[predictors])
        
        return pd.concat([X_w_NAs,X_wo_NAs])