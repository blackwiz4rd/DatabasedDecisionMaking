import numpy as np
# model for fitting dataset
from sklearn.ensemble import RandomForestClassifier

class ProjectBanker:

    def __init__(self):
        self.name = 'project'

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    """
    This function uses a random forest classifier to predict new probabilities
    """
    def fit(self, X, y):
        self.data = [X, y]
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0) # storing classifier
        self.clf.fit(X, y)

    def test_accuracy(self, X, y):
        return self.clf.score(X, y)

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
        ## order is class 1 = good_loan_proba, 2 = bad_loan_proba
        # print("classes", self.clf.classes_)
        return self.clf.predict_proba(np.reshape(x.to_numpy(), (1, -1)))[0][1]

    # The expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    """
    FIX: Is it correct to calcule utility like this?
    This function returns the probable amount gained or lost
    when granting or not the loan.
    """
    def expected_utility(self, x, action):
        amount_of_loan = x['amount']
        length_of_loan = x['duration']
        if action == 1:
            return pow(amount_of_loan*(1 + self.rate), length_of_loan) * (1-self.predict_proba(x))

        return -amount_of_loan * self.predict_proba(x)
        ##

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    """
    This function returns the best action such that the expected utility is maximized
    """
    def get_best_action(self, x):
        actions = [0, 1]
        best_action = -np.inf
        best_utility = -np.inf
        for a in actions:
            utility_a = self.expected_utility(x, a)
            if utility_a > best_utility:
                best_action = a
                best_utility = utility_a
        return best_action
