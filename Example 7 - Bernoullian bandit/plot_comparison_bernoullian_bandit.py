import pandas as pd
import matplotlib.pyplot as plt


# learning_policy="Thompson_sampling"

learning_policies = ["Greedy", r"Constant-$\epsilon$-greedy", r"$\epsilon$-greedy", "Boltzmann exploration", "Upper Confidence Bound", "Thompson Sampling", "Thompson Sampling greedy action selection"]
for learning_policy in learning_policies:
    print(learning_policy)
    if learning_policy == "Greedy":
        probabilities_dataframe = pd.read_csv("data/results_with_greedy_naive_update.csv")
    elif learning_policy == r"Constant $\epsilon$": 
        probabilities_dataframe = pd.read_csv(r"data/results_with_constant_epsilon_naive_update.csv")
    elif learning_policy == r"$\epsilon$ greedy": 
        probabilities_dataframe = pd.read_csv(r"data/results_with_epsilon_greedy_naive_update.csv")
    elif learning_policy == "Boltzmann exploration":
        probabilities_dataframe = pd.read_csv(r"data/results_with_boltzmann_exploration_naive_update.csv")
    elif learning_policy == "Upper Confidence Bound":
        probabilities_dataframe = pd.read_csv(r"data/results_with_upper_confidence_bound.csv")
    elif learning_policy == "Thompson Sampling":
        probabilities_dataframe = pd.read_csv(r"data/results_with_thompson_sampling.csv")
    elif learning_policy == "Thompson Sampling greedy action selection":
        probabilities_dataframe = pd.read_csv(r"data/results_with_thompson_sampling_greedy.csv")


    real_probabilities = probabilities_dataframe.columns
    print(probabilities_dataframe.shape)
    print(probabilities_dataframe)

    fig = plt.figure(figsize=(8,6))
    plt.plot(probabilities_dataframe.iloc[:, 0])
    plt.plot(probabilities_dataframe.iloc[:, 1])
    plt.plot(probabilities_dataframe.iloc[:, 2])
    plt.ylim([0, 1])
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Number of episodes", fontsize = 14)
    plt.ylabel("Probability of choosing the action", fontsize = 14)
    plt.title(learning_policy)
    plt.legend([r"$\theta_0 = $"+ str(real_probabilities[0]) , r"$\theta_1 = $" + str(real_probabilities[1]), r"$\theta_2 = $" + str(real_probabilities[2])])
    plt.show()



