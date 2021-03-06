import numpy as np
import pandas as pd
import numpy.random as npr
# model for fitting dataset
# implement a nn here with keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils import class_weight # assign a class weight
# feature selection
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel

# suppress warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class ProjectBanker:

    def __init__(self):
        self.name = 'nn'
        npr.seed(100)

    def preprocessing(self, X, fit=False):
        # try:
        #     X_temp = X.copy()[self.new_features]
        # except:
        #     X_temp = X.copy()[self.sorting_indexes]
        X_temp = X.copy()

        if fit:
            self.scaler = MinMaxScaler()
            X_some_features = self.scaler.fit_transform(X_temp)
        else:
            self.scaler.transform(X_temp)

        return X_temp

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    """
    This function uses a neural network classifier to predict new probabilities
    """
    def fit(self, X, y):
        y = y - 1 # 0 -> 1 good loan, 1 -> 2 bad loan
        # feature selection
        # clf = ExtraTreesClassifier(n_estimators=100)
        # clf = clf.fit(X, y)
        # self.sorting_indexes = np.argsort(clf.feature_importances_)
        # self.new_features = [X.columns.values[i] for i in sorting_indexes][28:]

        X_scaled = self.preprocessing(X, fit=True)

        ## nn with keras
        self.model = Sequential([
            Dense(64, input_shape=(X_scaled.shape[1],)),
            Activation('tanh'),
            Dense(32),
            Activation('tanh'),
            Dense(16),
            Activation('tanh'),
            Dense(1),
            Activation('sigmoid'),
        ])

        # class_weights = class_weight.compute_class_weight(
        #     'balanced',
        #     np.unique(y),
        #     y
        # ) # 0 -> 1 good loan, 1 -> 2 bad loan
        class_weights = {0: 700, 1:300}

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # print(self.model.summary())
        # test 150
        self.model.fit(X_scaled, y, epochs=20)

    def test_accuracy(self, X, y):
        y = y - 1
        X_scaled = self.preprocessing(X)
        test_loss, test_acc = self.model.evaluate(X_scaled, y)
        return test_acc

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
        # preprocessing
        x_scaled = self.preprocessing(x_reshaped)
        prediction = self.model.predict(x_scaled)
        return prediction[0][0]

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
    amount_of_loan * (pow(1 + self.rate, length_of_loan)) * (1 - self.predict_proba(x)) +
    -amount_of_loan * self.predict_proba(x)
    The expected utility if the action is not to grant anything is: 0
    This is true because we don't loose or get anything.
    """
    def expected_utility(self, x, action):
        amount_of_loan = x['amount']
        length_of_loan = x['duration']
        if action == 1:
            proba = self.predict_proba(x)
            gain = amount_of_loan * (pow(1 + self.rate, length_of_loan)) * (1 - proba)
            loss = amount_of_loan * proba
            return gain - loss

        return 0

    def expected_fairness(self, action, sensitive):
        p_good = np.array([0.713985, 0.428571]) # from whole dataset
        p_bad = np.ones(p_good.size) - p_good
        # good if action == 1 == grant
        p = p_good if action == 1 else p_bad
        return p[sensitive]

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    """
    This function returns the best action such that the expected utility is
    maximized
    """
    def get_best_action(self, x, lam=0):
        actions = [0, 1]
        sensitive = x['credit history_A31']
        utility_0 = (1-lam)*self.expected_utility(x, actions[0]) - lam*self.expected_fairness(actions[0], sensitive)
        utility_1 = (1-lam)*self.expected_utility(x, actions[1]) - lam*self.expected_fairness(actions[1], sensitive)
        if utility_1 > utility_0:
            return actions[1]

        return actions[0]
