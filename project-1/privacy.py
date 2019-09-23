import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import scipy.stats as stats

def set_range(X_col, step):
    return (X_col/step).round(1).astype(int)*step

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def select_features(X, encoded_features):
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X[encoded_features], X[target])
    # sort_feature = np.sort(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X[encoded_features])

    sorting_indexes = np.argsort(clf.feature_importances_)[-X_new.shape[1]:]
    new_features = [encoded_features[i] for i in sorting_indexes]
    print("new_features and accuracy", new_features)

    X_new = pd.DataFrame(X_new, columns=new_features)
    return X_new, new_features

def add_privacy(X, encoded_features, target, interest_rate, makePlots=False):

    X_privacy = X.copy()

    # select 20 most important features
    X_privacy, new_features = select_features(X_privacy, encoded_features)

    for i in encoded_features:
        print("feature ", i, np.unique(X[i]), min(X[i]), max(X[i]))

    # fitting gamma distribution to amount
    # plt.hist(X['amount'], bins=40, density=True)
    param = stats.gamma.fit(X['amount'], floc=0)
    x = np.linspace(0, X['amount'].max(), 1000)
    pdf_fitted = stats.gamma.pdf(x, *param)

    # setting the amount to the nearest value of the gamma distribution
    for i in range(X.shape[0]):
        X_privacy.loc[i, 'amount'] = find_nearest(pdf_fitted, X['amount'][i])

    if makePlots == True:
        plt.plot(x, pdf_fitted, color='r', label='fitting distribution')
        plt.hist(X['amount'], bins=40, density=True, label='histogram')
        plt.legend()
        plt.ylabel("count")
        plt.xlabel("amount")
        plt.show()

    print('true average amount', np.mean(X['amount']))
    epsilon = 0.1
    # local eps-privacy
    local_sensitivity = np.max(X['amount'])
    local_noise = np.random.laplace(scale=local_sensitivity / epsilon, size=X.shape[0])
    X_privacy['local_amount'] = X['amount'] + np.abs(local_noise)
    local_amount_err = np.abs(np.log(X['amount']/X_privacy['local_amount']))
    print('average amount with local DP + Laplace', np.mean(X_privacy['local_amount']))

    # central eps-privacy
    central_sensitivity = np.max(X['amount']) / X.shape[0]
    central_noise = np.random.laplace(scale=central_sensitivity / epsilon, size=1) # single value
    X_privacy['central_amount'] = X['amount'] + central_noise
    central_amount_err = np.abs(np.log(X['amount']/X_privacy['central_amount']))
    print('central_amount_err max mean', np.max(central_amount_err), np.mean(central_amount_err))
    print('average amount with central DP + Laplace', np.mean(X_privacy['central_amount']))
    # plt.plot(X_privacy['central_amount'], label='data+central')
    # plt.show()

    amount_err = np.abs(np.log(X['amount']/X_privacy['amount']))
    ## feature ranging
    X_privacy['duration'] = set_range(X['duration'], step=2)
    duration_err = np.abs(np.log(X_privacy['duration']/X['duration']))
    X_privacy['age'] = set_range(X['age'], step=10)
    age_err = np.abs(np.log(X_privacy['age']/X['age']))

    if makePlots == True:
        plt.title('ε-Differential Privacy')
        print(amount_err)
        plt.ylabel('ε')
        plt.boxplot([local_amount_err, central_amount_err, amount_err, duration_err, age_err], labels=['Local amount', 'Central amount', 'Amount error', 'Duration error', 'Age error'])
        plt.show()

    # restore to normal
    X_privacy['amount'] = X_privacy['central_amount']
    X_privacy.drop(columns=['central_amount', 'local_amount'])

    X_privacy['repaid'] = X['repaid']
    return X_privacy, new_features


import pandas
import sys
pandas.set_option('display.max_columns', None)

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../credit/german.data', sep=' ',
                     names=features+[target])
import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

## printing used data
print(X.info())
print(X[encoded_features].head())
print(X[target])
interest_rate = 0.05
X_privacy, encoded_features = add_privacy(X, encoded_features, target, interest_rate)
