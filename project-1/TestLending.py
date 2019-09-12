import pandas

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

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0

    ## This is to know how well the classifier is working
    ## ABOUT 70% of accuracy
    if decision_maker.name == "project":
        print("Testing accuracy of the classifier : ", decision_maker.test_accuracy(X_test, y_test))

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    action_results = []
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        action_results.append(action)
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan != 1):
                utility -= amount # number of credits lost
            else:
                utility += amount*(pow(1 + interest_rate, duration) - 1) # number of credits gained

    print("granted loans", np.array(action_results)[np.array(action_results)==1].sum())
    return utility


## Main code


### Setup model
import random_banker # this is a random banker
random_decision_maker = random_banker.RandomBanker()
import project_banker
decision_maker = project_banker.ProjectBanker()

interest_rate = 0.05 # r, if credit worthly insurer gets this amount per month

## printing used data
print(X[encoded_features].head())
print(X[target])

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
def get_utilities(X, encoded_features, target, interest_rate, decision_maker, n_tests=100):
    utility = []
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train, y_train)
        utility.append(test_decision_maker(X_test, y_test, interest_rate, decision_maker))
    return utility
    ### the std is an important measure when considering
    ### the random decision maker

import numpy as np
random_utility = get_utilities(X, encoded_features, target, interest_rate, random_decision_maker, n_tests=10)
# the objective is to increase this number
print("utility per tests on random decision maker, avg %i, std %i " % (np.mean(random_utility), np.std(random_utility)))

utility = get_utilities(X, encoded_features, target, interest_rate, decision_maker, n_tests=10)
# the objective is to increase this number
print("utility per tests on our decision maker, avg %i, std %i" % (np.mean(utility), np.std(utility)))

import matplotlib.pyplot as plt
x = [random_utility, utility]
plt.boxplot(x, labels=['random_utility', 'our_utility'])
plt.show()

## other interesting plots on random:
## best_action (it is a known distribution)
## out_utility depends on data so of course the distribution
## is different and variable for best_action
