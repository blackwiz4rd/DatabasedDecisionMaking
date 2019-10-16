import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from TestAuxiliary import *

# training on z
from sklearn.ensemble import RandomForestClassifier

# train to predict z based on X
def fit_sensitive(X, z):
    self.clf = RandomForestClassifier(
        n_estimators=100,
        random_state=0,
        max_depth=5,
        # max_features=15,
        class_weight="balanced"
    ) # storing classifier
    self.clf.fit(X, z)
    print("training score ", self.clf.score(X, z))

def predict_proba_sensitive(self, z):
    z_reshaped = np.reshape(z.to_numpy(), (1, -1))
    prediction = self.clf.predict_proba(z_reshaped)
    return prediction[0][1]

def calculate_proba(X, y):
    ef = X.columns[0]
    target = 'repaid'
    X[target] = y
    target_counts = y.value_counts()
    ef_vs_target = X[[ef, target]].groupby(ef).sum()
    ef_vs_target['tot'] = X[ef].value_counts()
    ef_vs_target[target] = ef_vs_target['tot'] - ef_vs_target[target]
    # y = 0 good loan
    ef_vs_target['prob'] = ef_vs_target[target]/ef_vs_target['tot'] # P(a|y=0,z)
    # ef_vs_target.drop('prob',axis=1).plot.bar()
    # plt.show()
    # deviation from the good loans probability
    ef_vs_target['deviation'] = np.abs(ef_vs_target['prob']-target_counts[0]/X.shape[0])
    # if ef == "age":
    #     plt.plot(ef_vs_target.index, ef_vs_target['prob'])
    #     plt.show()

    return ef_vs_target

def test_fairness(X, y):
    y = y - 1 # 1->0 good loan, 2->1 bad loan
    target_counts = y.value_counts()
    print("target counts ", target_counts)
    print("total", X.shape[0])

    all_deviations = []
    encoded_features = X.columns
    for ef in encoded_features:
        ef_vs_target = calculate_proba(X[[ef]], y)
        all_deviations.append(ef_vs_target['deviation'])
        print(ef_vs_target.to_latex())

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.boxplot(all_deviations)
    # ax.set_xticklabels(encoded_features, rotation=90)
    # plt.show()

    # plt.plot(np.unique(X['age']), all_deviations[encoded_features.index('age')])
    # plt.show()

    print(y.value_counts())

def main():
    pd.set_option('display.max_columns', None)

    X, encoded_features, target = dataset_setup()

    print(encoded_features)

    test_fairness(X[encoded_features], X[target])

    corr = X.corr()
    sns.heatmap(corr[corr>0.6])
    plt.show()

    # with pm.Model() as logistic_model:
    #     str_encoded_features = ''
    #     # for ef in encoded_features:
    #         # str_encoded_features += ef + ' + '
    #     # str_encoded_features = str_encoded_features[:-2]
    #     str_encoded_features = 'duration + amount + installment + age'
    #     print(str_encoded_features)
    #     pm.glm.GLM.from_formula(target + ' ~ ' + str_encoded_features, X, family = pm.glm.families.Binomial())
    #     trace = pm.sample(500, tune = 1000, init = 'adapt_diag')
    #
    # az.plot_trace(trace);

    # #Using Pearson Correlation
    # # plt.figure(figsize=(12,10))
    # # cor = X[encoded_features].corr()
    # # cor = cor[cor>0.2]
    # # sns.heatmap(cor, cmap=plt.cm.Reds)
    # # plt.show()
    # #
    # # sns.distplot(X['age'])
    # # plt.show()
    # #
    # # sns.jointplot(x=X["amount"], y=X["duration"]);
    # # plt.show()
    # #
    # # sns.pairplot(X[["amount", "duration"]])
    # # plt.show()

if __name__ == "__main__":
    main()
