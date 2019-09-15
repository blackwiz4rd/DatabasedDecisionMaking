import numpy as np

class RandomBanker:

    def __init__(self):
        self.name = 'random'

    def fit(self, X, y):
        self.data = [X, y]

    def set_interest_rate(self, rate):
        self.rate = rate
        # return

    def predict_proba(self, x):
        return 0

    def expected_utility(self, x, action):
        print("Expected utility: Not implemented")

    """
    The returned action is either to grant (1) or not the loan (0) randomly
    """
    def get_best_action(self, x):
        return np.random.choice(2,1)[0]
