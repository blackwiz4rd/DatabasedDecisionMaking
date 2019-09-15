import numpy as np

class DeterministicBanker:

    """
    Initialize banker with a given action: 0 don't grant, 1 grant the loan
    """
    def __init__(self, action):
        self.name = 'deterministic'
        self.action = action

    def fit(self, X, y):
        pass

    def set_interest_rate(self, rate):
        self.rate = rate

    def predict_proba(self, x):
        return 0


    def expected_utility(self, x, action):
        print("Expected utility: Not implemented")


    """
    Always grant or not grant the loan
    """
    def get_best_action(self, x):
        return self.action
