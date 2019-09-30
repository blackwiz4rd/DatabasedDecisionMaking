import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from TestAuxiliary import *
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
    # print("new_features and accuracy", new_features)

    X_new = pd.DataFrame(X_new, columns=new_features)
    return X_new, new_features

def add_privacy(X, encoded_features, target, interest_rate, makePlots=False, epsilon=0.1, local_privacy=True):
    print("----> adding privacy with epsilon ", epsilon)
    X_privacy = X.copy()

    # select 20 most important features
    # X_privacy, new_features = select_features(X_privacy, encoded_features)
    new_features = encoded_features

    for i in encoded_features:
        # print("feature ", i, np.unique(X[i]), min(X[i]), max(X[i]))
        pass

    binary_features = [
        'checking account balance_A12', 'checking account balance_A13', 'checking account balance_A14',
        'credit history_A31', 'credit history_A32', 'credit history_A33', 'credit history_A34',
        'purpose_A41', 'purpose_A410', 'purpose_A42', 'purpose_A43', 'purpose_A44', 'purpose_A45',
        'purpose_A46', 'purpose_A48', 'purpose_A49',
        'savings_A62', 'savings_A63', 'savings_A64', 'savings_A65',
        'employment_A72', 'employment_A73', 'employment_A74', 'employment_A75',
        'marital status_A92', 'marital status_A93', 'marital status_A94',
        'other debtors_A102', 'other debtors_A103',
        'property_A122', 'property_A123', 'property_A124',
        'other installments_A142', 'other installments_A143',
        'housing_A152', 'housing_A153',
        'job_A172', 'job_A173', 'job_A174',
        'phone_A192', 'foreign_A202',
        'persons'
    ]

    # credits, residence time, installment

    for b in binary_features:
        change = X_privacy.sample(frac = 0.5, random_state=42).index
        X_privacy.loc[change, b] = 1

    # print(X_privacy['phone_A192'])

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

    # print('true average amount', np.mean(X['amount']))
    # epsilon = 0.1
    # local eps-privacy
    local_sensitivity = np.max(X['amount'])
    local_noise = np.random.laplace(scale=local_sensitivity / epsilon, size=X.shape[0])
    X_privacy['local_amount'] = X['amount'] + np.abs(local_noise)
    local_amount_err = np.abs(np.log(X['amount']/X_privacy['local_amount']))
    # print('average amount with local DP + Laplace', np.mean(X_privacy['local_amount']))

    # central eps-privacy
    central_sensitivity = np.max(X['amount']) / X.shape[0]
    central_noise = np.random.laplace(scale=central_sensitivity / epsilon, size=1) # single value
    X_privacy['central_amount'] = X['amount'] + central_noise
    central_amount_err = np.abs(np.log(X['amount']/X_privacy['central_amount']))
    # print('central_amount_err max mean', np.max(central_amount_err), np.mean(central_amount_err))
    # print('average amount with central DP + Laplace', np.mean(X_privacy['central_amount']))
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
        # print(amount_err)
        plt.ylabel('ε')
        plt.boxplot([local_amount_err, central_amount_err, amount_err, duration_err, age_err], labels=['Local amount', 'Central amount', 'Amount error', 'Duration error', 'Age error'])
        plt.show()

    # restore to normal
    X_privacy['amount'] = X_privacy['local_amount'] if local_privacy == True else X_privacy['central_amount']
    X_privacy.drop(columns=['central_amount', 'local_amount'])

    X_privacy['repaid'] = X['repaid']
    return X_privacy, new_features

def main():
    pd.set_option('display.max_columns', None)

    ## Set up for dataset
    X, encoded_features, target = dataset_setup()

    ## printing used data
    # print(X.info())
    # print(X[encoded_features].head())
    # print(X[target])
    interest_rate = 0.05
    X_privacy, encoded_features = add_privacy(X, encoded_features, target, interest_rate)

if __name__ == "__main__":
    main()
