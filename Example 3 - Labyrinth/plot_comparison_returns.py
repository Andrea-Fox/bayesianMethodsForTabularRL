import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.series import Series


returns_sarsa = pd.read_csv("data/returns_1_step_sarsa.csv")
returns_q_learning = pd.read_csv("data/returns_q_learning.csv")
returns_expected_sarsa = pd.read_csv("data/returns_expected_sarsa.csv")

print(returns_q_learning.shape)
print(returns_sarsa.shape)

start_index = 9750
end_index = 10000

returns_q_learning = Series(returns_q_learning.iloc[:,1])
returns_sarsa = Series(returns_sarsa.iloc[:, 1])
returns_expected_sarsa = Series(returns_expected_sarsa.iloc[:, 1])

moving_average_q_learning = returns_q_learning.rolling(window = 500, min_periods= 100).mean()
moving_average_sarsa = returns_sarsa.rolling(window = 500, min_periods= 100).mean()
moving_average_expected_sarsa = returns_expected_sarsa.rolling(window = 500, min_periods= 100).mean()

print(moving_average_sarsa)

color_sarsa =  (158/256, 42/256, 43/256)
color_expected_sarsa = (158/256, 189/256, 110/256)
color_q_learning = (254/256, 168/256, 47/256)


plt.plot(moving_average_q_learning[:10000], color = color_q_learning)
plt.plot(moving_average_sarsa[:10000], color = color_sarsa)
# plt.plot(moving_average_expected_sarsa[:10000], color = color_expected_sarsa)
plt.rcParams.update({'font.size': 13})    
plt.xlabel("Number of episodes", fontsize = 13)
plt.ylabel("Reward", fontsize = 13)
plt.legend(["Q-learning", "Sarsa", "Expected Sarsa"])
plt.ylim([-100, 0])
plt.show()

plt.scatter(range(start_index, end_index), returns_q_learning[start_index:end_index], color = color_q_learning, s = 7)
plt.scatter(range(start_index, end_index),returns_sarsa[start_index:end_index], color = color_sarsa, s = 7)
# plt.scatter(range(start_index, end_index), returns_expected_sarsa[start_index:end_index], color = color_expected_sarsa, s = 7)
plt.rcParams.update({'font.size': 13})    
plt.xlabel("Number of episodes", fontsize = 13)
plt.ylabel("Reward", fontsize = 13)
plt.legend(["Q-learning", "Sarsa", "Expected Sarsa"])

plt.show()
# 