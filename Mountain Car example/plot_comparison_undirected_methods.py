import pandas as pd
import matplotlib.pyplot as plt

dataframe_comparison = pd.read_csv("data/risultati_numerici/comparison_rewards_undirected_mountain_car.csv")

exploration_types = ["greedy", "constant-epsilon", "epsilon-greedy", "boltzmann-exploration"]



fig = plt.figure(figsize=(8, 6))
for i in range(dataframe_comparison.shape[1]):
    plt.plot(pd.Series(dataframe_comparison.iloc[:, i]).rolling(100, min_periods = 10).mean())
    
plt.ylim([0, 210])
plt.rcParams.update({'font.size': 16})    
plt.xlabel("Number of episodes", fontsize = 16)
plt.ylabel("Reward", fontsize = 16)
plt.title("Undirected methods", fontsize = 20)
plt.legend(exploration_types)
plt.show()