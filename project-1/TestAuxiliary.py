import pandas as pd
import numpy as np
from privacy import add_privacy
from sklearn.model_selection import train_test_split

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker, bootstrap=False):
    n_test_examples = X_test.shape[0]
    utility = 0

    ## This is to know how well the classifier is working
    ## ABOUT 70% of accuracy
    if decision_maker.name == "forest" or decision_maker.name == "nn":
        # bootstrapping to test accuracy
        if bootstrap == True:
            n_bootstrap_samples = 1000
            bootstrap_test_score = np.zeros(n_bootstrap_samples)

            for t in range(n_bootstrap_samples):
                bootstrap_X_test, bootstrap_y_test = resample(X_test, y_test, replace=True, n_samples = X_test.shape[0])
                bootstrap_test_score[t] = decision_maker.test_accuracy(bootstrap_X_test, bootstrap_y_test)
            plt.hist(bootstrap_test_score, bins=40)
            plt.title("Bootstrapped test scores")
            plt.show()
            print("Bootstrap testing mean accuracy of the classifier : ", bootstrap_test_score.mean())

        print("Single test accuracy of the classifier : ", decision_maker.test_accuracy(X_test, y_test))

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    action_results = np.array([])
    for t in range(n_test_examples):
        if decision_maker.name == "perfect":
            action = decision_maker.get_best_action(y_test.iloc[t])
        else:
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

### Do a number of preliminary tests by splitting the data in parts
def get_utilities(X, encoded_features, target, interest_rate, decision_maker, n_tests, epsilons):
    set_privacy = epsilons.size != 0

    utility = []
    # do this once just for the random_forest to get the best value of depth
    print("-- Running banker:", decision_maker.name)
    for iter in range(n_tests):
        # set different privacy with epsilon on each test
        if set_privacy == True:
            central_privacy = True
            X_temp, encoded_features_temp = add_privacy(X, encoded_features, target, interest_rate, epsilon=epsilons[iter], local_privacy=local_privacy)
            X_train, X_test, y_train, y_test = train_test_split(X_temp[encoded_features_temp], X_temp[target], test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2, random_state=iter)
        # random_state=iter is mandatory otherwise we get wrong results
        # splitting the set would be different for each model

        ## preprocessing on random forest
        if decision_maker.name == "forest":
            decision_maker.set_best_max_depth(X_train, y_train)
            decision_maker.set_best_max_features(X_train, y_train)

        decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train, y_train)
        u = test_decision_maker(X_test, y_test, interest_rate, decision_maker)
        print("utility achieved: ", u)
        utility.append(u)
    return utility

def dataset_setup():
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign']
    target = 'repaid'
    df = pd.read_csv('../credit/german.data', sep=' ',
                         names=features+[target])

    numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
    quantitative_features = list(filter(lambda x: x not in numerical_features, features))
    X = pd.get_dummies(df, columns=quantitative_features, drop_first=True)
    encoded_features = list(filter(lambda x: x != target, X.columns))
    return X, encoded_features, target
