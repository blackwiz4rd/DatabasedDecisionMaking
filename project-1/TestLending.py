import pandas as pd
import numpy as np
import sys
from sklearn.utils import resample
import matplotlib.pyplot as plt
from TestAuxiliary import *

# custom setting to display all columns
pd.set_option('display.max_columns', None)

## Set up for dataset
X, encoded_features, target = dataset_setup()

## Main code

### Setup model
import random_banker # this is a random banker
import project_banker
import deterministic_banker
import nn_banker
import perfect_banker

## initializing all the random bankers
random_decision_maker = random_banker.RandomBanker()
decision_maker = project_banker.ProjectBanker()
deterministic_grant_banker = deterministic_banker.DeterministicBanker(action=1)
deterministic_nogrant_banker = deterministic_banker.DeterministicBanker(action=0)
nn_banker = nn_banker.ProjectBanker()
perfect_banker = perfect_banker.PerfectBanker()

interest_rate = 0.05 # r value

## printing used data
# print(X.info())
# print(X[encoded_features].head())
# print(X[target])

if len(sys.argv) == 2:
    n_tests = int(sys.argv[1])
else:
    n_tests = 100

# set to true if using privacy
set_privacy = True
# range from 0.1 to 100
if set_privacy == True:
    epsilons = np.linspace(start=0.1,stop=30,num=n_tests)


print("true values on dataset for: granted loans, not granted loans", np.sum(X[target]==1), np.sum(X[target]==2))

# utility = np.zeros(len(random_utility))
utility = get_utilities(X, encoded_features, target, interest_rate, decision_maker, n_tests, epsilons)
print("utility per tests with random forest, avg %i, std %i" % (np.mean(utility), np.std(utility)))

# nn_utility = np.zeros(len(utility))
nn_utility = get_utilities(X, encoded_features, target, interest_rate, nn_banker, n_tests, epsilons)
print("utility per tests on nn, avg %i, std %i" % (np.mean(nn_utility), np.std(nn_utility)))

random_utility = get_utilities(X, encoded_features, target, interest_rate, random_decision_maker, n_tests, epsilons)
print("utility per tests on random decision maker, avg %i, std %i " % (np.mean(random_utility), np.std(random_utility)))

deterministic_grant_utility = get_utilities(X, encoded_features, target, interest_rate, deterministic_grant_banker, n_tests, epsilons)
print("utility per tests on granting always, avg %i, std %i" % (np.mean(deterministic_grant_utility), np.std(deterministic_grant_utility)))

deterministic_nogrant_utility = get_utilities(X, encoded_features, target, interest_rate, deterministic_nogrant_banker, n_tests, epsilons)
print("utility per tests on not granting always, avg %i, std %i" % (np.mean(deterministic_nogrant_utility), np.std(deterministic_nogrant_utility)))

perfect_utility = get_utilities(X, encoded_features, target, interest_rate, perfect_banker, n_tests, epsilons)
print("utility per tests on perfect banker, avg %i, std %i" % (np.mean(perfect_utility), np.std(perfect_utility)))

utilities = [random_utility, utility, nn_utility, deterministic_grant_utility, deterministic_nogrant_utility, perfect_utility]
labels=['rand', 'forest', 'nn', 'grant', 'nogrant', 'perfect']

## plots

if set_privacy == True:
    for u, l in zip(utilities, labels):
        plt.plot(epsilons, u, label=l)
        plt.ylabel("utility")
        plt.xlabel("epsilon")
        plt.legend()
    plt.show()
else:
    plt.boxplot(utilities, labels=labels)
    plt.ylabel("utility")
    plt.show()

for u, l in zip(utilities, labels):
    plt.plot(range(len(u)), u, '.', label=l, alpha=0.5)
plt.yscale("log")
plt.legend()
plt.ylabel("utility")
plt.xlabel("test number")
plt.show()

gainable_col = X['amount']*(pow(1 + interest_rate, X['duration']) - 1)
gainable_col[X[target] == 2] = 0
plt.hist(gainable_col, bins=40)
plt.ylabel("count")
plt.xlabel("gainable amount")
plt.show()
