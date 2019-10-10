
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from TestAuxiliary import *
from scipy import stats

import project_banker

def main():
    pd.set_option('display.max_columns', None)

    ## Set up for dataset
    X, encoded_features, target = dataset_setup()

    ## printing used data
    # print(X.info())
    # print(X[encoded_features].head())
    # print(X[target])
    # print(encoded_features)


    # testing on a singlel model
    decision_maker = project_banker.ProjectBanker()
    interest_rate = 0.05
    X_test, y_test, a_results = get_test_predictions(X, encoded_features, target, interest_rate, decision_maker, set_privacy=False)

    # for i in encoded_features:
    #     print("feature ", i, np.unique(X[i]), min(X[i]), max(X[i]))

    # singles
    print("number of singles", X_test[X_test['persons']==1].shape[0], "number of not singles", X_test[X_test['persons']==2].shape[0])
    print("results on number of singles", a_results[X_test['persons']==1], a_results[X_test['persons']==2])

    # count granted loans based on duration

    #Using Pearson Correlation
    # plt.figure(figsize=(12,10))
    # cor = X[encoded_features].corr()
    # cor = cor[cor>0.2]
    # sns.heatmap(cor, cmap=plt.cm.Reds)
    # plt.show()
    #
    # sns.distplot(X['age'])
    # plt.show()
    #
    # sns.jointplot(x=X["amount"], y=X["duration"]);
    # plt.show()
    #
    # sns.pairplot(X[["amount", "duration"]])
    # plt.show()

if __name__ == "__main__":
    main()
