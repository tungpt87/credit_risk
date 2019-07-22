import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


#deprecated    
def plotROC_AUC(target,pred):
    from sklearn import metrics
    metrics.roc_auc_score(target,pred)

    fpr, tpr, _ = metrics.roc_curve(target, pred)
    roc_auc = metrics.auc(fpr, tpr)
    #xgb.plot_importance(gbm)
    #plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()




#deprecated
def plotHistNumerical(data, columns, hue = None, bins = 50, col_wrap=5):
    import seaborn as sns
    from matplotlib import pyplot as plt
    # Plot histograms of numerical columns
    melted = None
    
    if hue is None:
        melted = data[columns].melt(var_name='cols',value_name='vals')
    else:
        melted = data[columns + [hue]].melt(hue,var_name='cols',value_name='vals')

    melted['vals'] = pd.to_numeric(melted['vals'].tolist())

    g = sns.FacetGrid(melted,col='cols',col_wrap=5,sharex=False,sharey=False,legend_out=True, hue = hue)
    g.map(plt.hist,"vals",bins=50)
    return g

#deprecated
def plotHistNumericalUnivariate(data,num_cols=None,hue=None,bins=50,col_wrap=5):
    # Plot histograms of numerical columns
    import seaborn as sns
    from matplotlib import pyplot as plt
    melted = None
    if hue is not None:
        melted = data[num_cols+[hue]].melt(hue,var_name='cols',value_name='vals')
    else:
        melted = data[num_cols].melt(var_name='cols',value_name='vals')

    melted['vals'] = pd.to_numeric(melted['vals'].tolist())

    g = sns.FacetGrid(melted,col='cols',col_wrap=col_wrap,sharex=False,sharey=False,legend_out=True,hue=hue)
    g.map(plt.hist,"vals",bins=bins)
    
    return g

#deprecated
def plotBoxPlot(data, columns, label = None, col_wrap = 5):
    import seaborn as sns
    from matplotlib import pyplot as plt
    # Plot histograms of numerical columns across labels

    # Plot histograms of numerical columns
    melted = None
    
    if label is not None:
        melted = data[columns+[label]].melt(label,var_name='cols',value_name='vals')
    else: 
        melted = data[columns].melt(var_name='cols',value_name='vals')

    g = sns.FacetGrid(melted,col='cols',col_wrap=5,sharex=False,sharey=False,legend_out=True)
    g.map(sns.boxplot,label,'vals').add_legend()
    
    return g

#deprecated
def imputeMissingValues(X, target, predictors = None, method = 'knn-reg', parameters = {'k':3}):
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    
    X_wo_NAs = X[~X[target].isna()]
    X_w_NAs = X[X[target].isna()]
    
    if predictors is None:
        predictors = [col for col in W_wo_NAs if col != target]
    
    algo = None
    
    if method == 'knn-reg':
        algo = KNeighborsRegressor(parameters['k'], weights = 'distance')
    if method == 'knn-cls':
        algo = KNeighborsClassifier(parameters['k'], weights = 'distance')
    else:
        pass
    
    model = algo.fit(X_wo_NAs[predictors], X_wo_NAs[target])
    
    X_w_NAs[target] = model.predict(X_w_NAs[predictors])
    
    return pd.concat([X_w_NAs,X_wo_NAs])
    
def logTransform(data, columns, eps = 1, normalization = True):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    
    temp = pd.DataFrame()
    for col in data.columns:
        if col in columns:
            temp[col] = np.log(data[col] + eps)
            
            if normalization:
                temp[col] = (temp[col] - np.min(temp[col]))/(np.max(temp[col]) - np.min(temp[col]))
        else:
            temp[col] = data[col]
    
    return temp


def splitTrainTest(data, train_proportion = 0.8, target = None, balance_test_set = False, seed = 0):
    '''
    Split a dataset into and train and test sets

    Args: 
    data: A Pandas dataframe
    train_proportion: (float or int) Desired proportion of training data. 
    target: Name of the target column. Required if balance_test_set = True
    balance_test_set: False by default.
    seed: Numpy's seed for random generator

    Return: Tuple of (train, test)    
    '''

    if (target is not None) and balance_test_set:
        trains = []
        tests = []

        # Find label with the least number of samples
        lbl = data.groupby(target).size().reset_index().sort_values(0)[target].tolist()[0]
        temp_train, temp_test = splitData(data[data['label'] == lbl], train_proportion, seed=seed)
        trains.append(temp_train)
        tests.append(temp_test)

        # Get the test size of 1 label and apply to the rest of the data
        test_size = len(temp_test)

        # Get the list of all label under "target" except lbl
        labels = [x for x in set(data[target].tolist()) if x != lbl]

        for label in labels:
            label_data = data[data['label'] == label]
            size = len(label_data)
            temp_train, temp_test = splitData(label_data, size - test_size, seed=seed)
            trains.append(temp_train)
            tests.append(temp_test)
        return pd.concat(trains, axis=0), pd.concat(tests, axis=0)
    else:
        return splitData(data, train_portion, seed = seed)


def splitData(data, train_proportion = 0.8, seed = 0):
    '''
    Randomly split a dataset into train and test sets

    Args:
    data: A Pandas dataframe
    train_proportion: (float or int) Desired proportion of training data. 
    seed: Numpy random generator's seed
    '''
    import numpy as np
    np.random.seed(seed)

    if train_proportion < 1:
        mask = np.random.rand(len(data)) < train_proportion
        train = data[mask]
        test = data[~mask]
    else:
        mask = np.random.rand(len(data)) * len(data) < train_proportion
        train = data[mask]
        test = data[~mask]
    
    return train, test





