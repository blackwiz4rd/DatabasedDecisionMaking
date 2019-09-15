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

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0

    ## This is to know how well the classifier is working
    ## ABOUT 70% of accuracy
    if decision_maker.name == "forest" or decision_maker.name == "nn":
        print("Testing accuracy of the classifier : ", decision_maker.test_accuracy(X_test, y_test))

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    action_results = np.array([])
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        action_results = np.append(action_results, action)
        good_loan = y_test.iloc[t] # assume the labels are correct
        # print("loan true value ", good_loan)
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan != 1):
                utility -= amount # number of credits lost
            else:
                utility += amount*(pow(1 + interest_rate, duration) - 1) # number of credits gained

    print(decision_maker.name, "granted loans", np.sum(action_results))
    return utility


## Main code


### Setup model
import random_banker # this is a random banker
random_decision_maker = random_banker.RandomBanker()
import project_banker
decision_maker = project_banker.ProjectBanker()
import deterministic_banker
deterministic_grant_banker = deterministic_banker.DeterministicBanker(action=1)
deterministic_nogrant_banker = deterministic_banker.DeterministicBanker(action=0)
import nn_banker
nn_banker = nn_banker.ProjectBanker()

interest_rate = 0.05 # r value

## printing used data
# print(X.info())
# print(X[encoded_features].head())
# print(X[target])

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
def get_utilities(X, encoded_features, target, interest_rate, decision_maker):
    if len(sys.argv) == 2:
        n_tests = int(sys.argv[1])
    else:
        n_tests = 100
    utility = []
    # do this once just for the random_forest to get the best value of depth
    print("-- Running banker:", decision_maker.name)
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2, random_state=iter)
        # random_state=iter is mandatory otherwise we get wrong results
        # splitting the set would be different for each model

        ## preprocessing on random forest
        if decision_maker.name == "forest":
            decision_maker.set_best_max_depth(X_train, y_train)

        decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train, y_train)
        utility.append(test_decision_maker(X_test, y_test, interest_rate, decision_maker))
    return utility

import numpy as np
random_utility = get_utilities(X, encoded_features, target, interest_rate, random_decision_maker)
print("utility per tests on random decision maker, avg %i, std %i " % (np.mean(random_utility), np.std(random_utility)))

# utility = np.zeros(len(random_utility))
utility = get_utilities(X, encoded_features, target, interest_rate, decision_maker)
print("utility per tests with random forest, avg %i, std %i" % (np.mean(utility), np.std(utility)))

# nn_utility = np.zeros(len(utility))
nn_utility = get_utilities(X, encoded_features, target, interest_rate, nn_banker)
print("utility per tests on nn, avg %i, std %i" % (np.mean(nn_utility), np.std(nn_utility)))

deterministic_grant_utility = get_utilities(X, encoded_features, target, interest_rate, deterministic_grant_banker)
print("utility per tests on granting always, avg %i, std %i" % (np.mean(deterministic_grant_utility), np.std(deterministic_grant_utility)))

deterministic_nogrant_utility = get_utilities(X, encoded_features, target, interest_rate, deterministic_nogrant_banker)
print("utility per tests on not granting always, avg %i, std %i" % (np.mean(deterministic_nogrant_utility), np.std(deterministic_nogrant_utility)))

import matplotlib.pyplot as plt
utilities = [random_utility, utility, nn_utility, deterministic_grant_utility, deterministic_nogrant_utility]
labels=['random', 'our', 'nn', 'deterministic_grant', 'deterministic_nogrant']

plt.boxplot(utilities, labels=labels)
plt.ylabel("utility")
plt.show()

for u, l in zip(utilities, labels):
    plt.plot(range(len(u)), u, '.', label=l, alpha=0.5)
plt.legend()
plt.ylabel("utility")
plt.xlabel("test number")
plt.show()

## plot some stuff
rate_column = X['amount']*(pow(1 + interest_rate, X['duration']) - 1)
plt.hist(rate_column)
plt.ylabel("count")
plt.xlabel("gainable amount")
plt.show()
