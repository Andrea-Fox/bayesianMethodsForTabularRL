import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


results_q_learning = pd.read_csv("data/result_q_learning.csv")
results_double_q_learning = pd.read_csv("data/result_double_q_learning.csv")


color_q_learning = (254/256, 168/256, 47/256)
color_double_q_learning = (84/256, 94/256, 117/256)

plt.plot(results_q_learning.iloc[:, 1], color = color_q_learning)
plt.plot(results_double_q_learning.iloc[:, 1], color = color_double_q_learning)
plt.hlines(0.05, xmin = -50, xmax = 1100, alpha= 0.65, color = 'grey', linestyles='dashed', label='' )
plt.ylim(0, 1)
plt.xlim(0, 300)
plt.xlabel("Number of episodes", fontsize = 12)
plt.ylabel("Percentage of left actions", fontsize = 12)
plt.legend(["Q-learning", "Double Q-learning", "Optimal value"])
plt.show()

