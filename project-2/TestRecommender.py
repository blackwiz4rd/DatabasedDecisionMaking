import numpy as np
import pandas as pd

import data_generation
import random_recommender
# import reference_recommender

def default_reward_function(action, outcome):
    return -0.1 * (action!= 0) + outcome

def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u

def single_test(data_generation, features, actions, outcome, policy_factory, matrices, n_tests = 1000):
    print("Setting up simulator")
    generator = data_generation.DataGenerator(matrices=matrices)
    print("Setting up policy")
    policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
    ## Fit the policy on historical data first
    print("Fitting historical data to the policy")
    policy.fit_treatment_outcome(features, actions, outcome)
    ## Run an online test with a small number of actions
    print("Running an online test")
    result = test_policy(generator, policy, default_reward_function, n_tests)
    print("Total reward:", result)
    print("Final analysis of results")
    policy.final_analysis()

if __name__ == "__main__" :
    np.random.seed(100)
    features = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ")
    features_labels = ['sex', 'smoker']
    features_labels.extend(['gene_expression_%i' % i for i in range(3,129)])
    features_labels.extend(['symptom_0', 'symptom_1'])
    features.columns = features_labels
    # therapeutic intervention: true = give intervention, false = don't give it
    actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
    # therapeutic intervention: true = alive, false = dead
    outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
    # observations = features.values[:, :128] # purpose?
    # labels = features.values[:,128] + features.values[:,129]*2 # purpose?
    print(features)

    policy_factory = random_recommender.RandomRecommender
    # policy_factory = reference_recommender.HistoricalRecommender

    ## First test with the same number of treatments
    print("---- Testing with only two treatments ----")
    single_test(data_generation, features, actions, outcome, policy_factory, "./generating_matrices.mat")

    ## First test with the same number of treatments
    print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
    single_test(data_generation, features, actions, outcome, policy_factory, "./big_generating_matrices.mat")
