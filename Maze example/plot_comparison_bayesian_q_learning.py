import pandas as pd
import matplotlib.pyplot as plt

dataframe_comparison = pd.read_csv("data/comparison_rewards_bayesian_q_learning_final.csv")


exploration_types = ["Greedy selection", "Q value sampling", "Myopic VPI selection"]



fig = plt.figure(figsize=(8, 6))
for i in range(dataframe_comparison.shape[1]):
    plt.plot(pd.Series(dataframe_comparison.iloc[:, i]).rolling(25, min_periods = 10).mean())
    
plt.ylim([-100, 30])
plt.rcParams.update({'font.size': 16})    
plt.xlabel("Number of episodes", fontsize = 16)
plt.ylabel("Reward", fontsize = 16)
plt.title("Bayesian Q-learning", fontsize = 20)
plt.legend(exploration_types)
plt.show()