import pandas as pd
import matplotlib.pyplot as plt

dataframe_comparison = pd.read_csv("data/comparison_rewards_undirected_methods.csv")
# dataframe_comparison = pd.read_csv("/home/andrea/Desktop/tesiReinforcmentLearning/codice/chain/risultati_numerici/comparison_epsilon_greedy.csv")




# exploration_rates = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4 ]
# exploration_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
exploration_types = ["greedy", "constant-epsilon", "epsilon-greedy", "boltzmann-exploration"]



fig = plt.figure(figsize=(8, 6))
for i in range(dataframe_comparison.shape[1]):
    plt.plot(pd.Series(dataframe_comparison.iloc[:, i]).rolling(25, min_periods = 10).mean())
    
plt.ylim([-100, 30])
plt.rcParams.update({'font.size': 16})    
plt.xlabel("Number of episodes", fontsize = 16)
plt.ylabel("Reward", fontsize = 16)
plt.title("Undirected methods", fontsize = 20)
plt.legend(exploration_types)
plt.show()