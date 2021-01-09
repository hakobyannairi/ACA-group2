import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

apartments = pd.read_csv('houses_train.csv')


def data_preprocessing(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    '''Takes care of numeric outliers and generates One-hot Encoding for categorical variables'''

    train.drop('Unnamed: 0', axis=1, inplace=True)

    try:
        test.drop('Unnamed: 0', axis=1, inplace=True)
    except:
        pass

    ## Numerical ##
    train_numeric = train.select_dtypes(include=['number'])
    train_numeric = train_numeric.reset_index()
    train_numeric['index'] = train_numeric['index'].apply(lambda x: str(x)+'_train')

    test_numeric = test.select_dtypes(include=['number'])
    test_numeric = test_numeric.reset_index()
    test_numeric['index'] = test_numeric['index'].apply(lambda x: str(x)+'_test')

    apartments_total_numeric = train_numeric.append(test_numeric)


    ## Taking care of numerical outliers
    iqr = apartments_total_numeric.describe().loc['75%',:] - apartments_total_numeric.describe().loc['25%',:]
    lower_bounds = apartments_total_numeric.describe().loc['25%',:] - 2.5 * iqr
    lower_bounds = lower_bounds.apply(lambda x: 0 if x<0 else x)
    lower_bounds.loc['price'] = 10000
    lower_bounds.loc['max_floor'] = 1
    lower_bounds.loc['area'] = 5
    lower_bounds.loc['num_rooms'] = 1


    upper_bounds = apartments_total_numeric.describe().loc['75%',:] + 8 * iqr
    upper_bounds.loc['num_bathrooms'] = 6
    upper_bounds.loc['ceiling_height'] = 6
    upper_bounds.loc['price'] = 900000 # according to officials the most expensive house in Yerevan is over 800K not reaching 900k

    indices_to_drop = []
    for col in apartments_total_numeric.drop(['index'],axis=1).columns:
        if not apartments_total_numeric[apartments_total_numeric[col] < lower_bounds[col]].empty:
            indices_to_drop += [apartments_total_numeric[apartments_total_numeric[col] > upper_bounds[col]].index]
        if not apartments_total_numeric[apartments_total_numeric[col] > upper_bounds[col]].empty:
            indices_to_drop += [apartments_total_numeric[apartments_total_numeric[col] > upper_bounds[col]].index]


    apartments_total_numeric = apartments_total_numeric.drop(indices_to_drop).reset_index(drop=True)


    ## Categorical ##
    train_object = train.select_dtypes(include=['object'])
    train_object = train_object.reset_index()
    train_object['index'] = train_object['index'].apply(lambda x: str(x)+'_train')

    test_object = test.select_dtypes(include=['object'])
    test_object = test_object.reset_index()
    test_object['index'] = test_object['index'].apply(lambda x: str(x)+'_test')


    apartments_total_object = train_object.append(test_object)
    apartments_total_object = apartments_total_object.drop(['url'], axis=1) # there was only 1 website not significant


    for col in apartments_total_object.columns:
        apartments_total_object[col] = apartments_total_object[col].apply(lambda x: x.lower())

    apartments_total_object['street'] = apartments_total_object['street'].apply(lambda x: x.split(' ')[0])

    apartments_total_object_onehot = pd.concat([apartments_total_object[['index']], pd.get_dummies(apartments_total_object[['condition','district','street','region','building_type']])], axis=1).reindex(apartments_total_object[['index']].index)
    apartments_preprocessed_total = pd.merge(apartments_total_numeric, apartments_total_object_onehot, how='left', on = 'index')
    apartments_preprocessed_total['index'] = apartments_preprocessed_total['index'].apply(lambda x: x.split('_')[-1])



    return (apartments_preprocessed_total[apartments_preprocessed_total['index'] == 'train'].drop(['index'], axis=1), apartments_preprocessed_total[apartments_preprocessed_total['index'] == 'test'].drop(['index'], axis=1))


def yerevan_houses_model(test: pd.DataFrame) -> np.ndarray:
    ''' Trains the model based off of our train and returns the predictions'''
    global apartments

    apartments_train = apartments #pd.read_csv('houses_train.csv')
    apartments_test = test

    pre_train, pre_test = data_preprocessing(apartments_train, apartments_test)

    X_train = pre_train.drop(['price'], axis=1)
    y_train = pre_train['price']
    X_test = pre_test.drop(['price'], axis=1)
    y_test = pre_test['price']

    # use polynomial transformation
    poly = PolynomialFeatures(2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)


    # Apply Ridge, the best score happens when the alphas are from 300-500
    alphas = np.linspace(1,500)
    clf = RidgeCV(alphas=alphas )
    clf.fit(X_train, y_train)
    print('Score R2: ', r2_score(clf.predict(X_test),y_test))


    return clf.predict(X_test)


if __name__ == '__main__':
    ## Please, input your train data here in place of test 
    
    # train = apartments.loc[:4500,:] #input your data 
    # test = apartments.loc[4501:,:]
    # apartments = train

    # Warning, the program will not work so, comment out the row below and uncomment the 3 lines above
    test = pd.read_csv('houses_train.csv') # input your name 
    predictions = yerevan_houses_model(test)
    
