import numpy as np
import pandas as pd
import numpy.random as npr
# model for fitting dataset
# implement a nn here with keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler

# suppress warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class ProjectBanker:

    def __init__(self):
        self.name = 'nn'
        npr.seed(100)

    def preprocessing(self, X, fit=False):
        X_temp = X.copy()
        # rescale_features = ['age', 'duration', 'amount']
        # X_some_features = X_temp[rescale_features]

        if fit:
            self.scaler = MinMaxScaler()
            X_some_features = self.scaler.fit_transform(X_temp)
        else:
            self.scaler.transform(X_temp)

        # X_temp[rescale_features] = X_some_features
        return X_temp

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    """
    This function uses a neural network classifier to predict new probabilities
    """
    def fit(self, X, y):
        print(X.shape, y.shape)
        y = y - 1
        X_scaled = self.preprocessing(X, fit=True)

        ## nn with keras
        self.model = Sequential([
            Dense(64, input_shape=(X.shape[1],)),
            Activation('relu'),
            Dense(32),
            Activation('relu'),
            Dense(16),
            Activation('relu'),
            Dense(1),
            Activation('sigmoid'),
        ])
        self.model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
        self.model.fit(X_scaled, y, epochs=7)

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

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    """
    This function returns the best action such that the expected utility is
    maximized
    """
    def get_best_action(self, x):
        actions = [0, 1]
        utility_0 = self.expected_utility(x, actions[0])
        utility_1 = self.expected_utility(x, actions[1])
        if utility_1 > utility_0:
            return actions[1]

        return actions[0]
