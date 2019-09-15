import numpy as np
import pandas as pd
# model for fitting dataset
from sklearn.ensemble import RandomForestClassifier
# select best model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class ProjectBanker:

    def __init__(self):
        self.name = 'forest'
        self.best_max_depth = None
        # self.best_max_depth = 10

    def preprocessing(self, X, fit=False):
        X_temp = X.copy()
        # rescale_features = ['age', 'duration', 'amount']
        # X_some_features = X_temp[rescale_features]

        if fit:
            self.scaler = StandardScaler()
            X_some_features = self.scaler.fit_transform(X_temp)
        else:
            self.scaler.transform(X_temp)

        # X_temp[rescale_features] = X_some_features
        return X_temp

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    """
    This function uses a random forest classifier to predict new probabilities
    """
    def fit(self, X, y):
        X_scaled = self.preprocessing(X)
        self.clf = RandomForestClassifier(n_estimators=100,  random_state=0, max_depth=self.best_max_depth) # storing classifier
        self.clf.fit(X_scaled, y)

    """
    Function invoked once to get the best max depth of the tree for
    the test set
    """
    def set_best_max_depth(self, X, y):
        if self.best_max_depth == None:
            X_scaled = self.preprocessing(X, fit=True)
            depths = range(5,20)
            untrained_models = [RandomForestClassifier(n_estimators=100, max_depth=d) for d in depths]
            fold_scores = [cross_val_score(estimator=m, X=X_scaled, y=y, cv=5) for m in untrained_models]
            mean_xv_scores = [s.mean() for s in fold_scores]
            self.best_max_depth = np.asarray(mean_xv_scores).argmax()

    """
    Function to test the accuracy of the classifier
    """
    def test_accuracy(self, X, y):
        X_scaled = self.preprocessing(X)
        return self.clf.score(X_scaled, y)

    # set the interest rate
    """
    This function stores the interest rate within the function
    """
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    """
    This function predicts the probability of failure (being a bad loan),
    given the data to predict for. It is necessary to cast the input
    to numpy since it is a Series.
    In case of single sample we also need to reshape it.
    """
    def predict_proba(self, x):
        x_reshaped = np.reshape(x.to_numpy(), (1, -1))
        x_scaled = self.preprocessing(x_reshaped)
        prediction = self.clf.predict_proba(x_scaled)
        return prediction[0][1]

    # The expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is
    # amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    """
    This function calculates the expected utility.
    The expected utility if the action is to grant the loan is given by
    the formula:
    amount_of_loan*(1 + self.rate)^length_of_loan * (1-self.predict_proba(x)) +
    -amount_of_loan * self.predict_proba(x)
    The expected utility if the action is not to grant anything is: 0
    This is true because we don't loose or get anything.
    """
    def expected_utility(self, x, action):
        amount_of_loan = x['amount']
        length_of_loan = x['duration']
        if action == 1:
            gain = amount_of_loan * (pow(1 + self.rate, length_of_loan)) * (1 - self.predict_proba(x))
            loss = amount_of_loan * self.predict_proba(x)
            return gain - loss

        return 0

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    """
    This function returns the best action such that the expected utility is
    maximized
    """
    def get_best_action(self, x):
        ## better way to calculate utility that allows to defiate from max
        ## threshold may be set higher to avoid granting too many loans
        actions = [0, 1]
        utility_0 = self.expected_utility(x, actions[0])
        utility_1 = self.expected_utility(x, actions[1])
        # grant about accuracy/100*200 = 150 -> error estimate
        # most of the measures are below 20 000
        if utility_1 - utility_0 > 1500:
            return actions[1]
        return actions[0]
