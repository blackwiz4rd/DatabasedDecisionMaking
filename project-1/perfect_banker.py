import numpy as np

class PerfectBanker:

    def __init__(self):
        self.name = 'perfect'

    def fit(self, X, y):
        pass

    def set_interest_rate(self, rate):
        self.rate = rate

    def predict_proba(self, x):
        return 0

    def expected_utility(self, x, action):
        print("Expected utility: Not implemented")

    """
    The returned action is either to grant (1) or not the loan (0),
    based on true value
    """
    def get_best_action(self, y):
        return 0 if y == 2 else y
