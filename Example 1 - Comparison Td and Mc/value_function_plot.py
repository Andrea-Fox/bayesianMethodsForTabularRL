import pandas as pd
import matplotlib.pyplot as plt


# scegliamo questi valori perché sono quelli che danno i risultati migliori

td_0_05 = pd.read_csv("data/temporal_difference_alpha_0-05.csv")
td_0_05 = td_0_05.to_numpy()

true_values = [1/6, 2/6, 3/6, 4/6, 5/6]

plt.plot(td_0_05[0, 2:], color = (63/256, 184/256, 175/256) )
plt.scatter(x = [0, 1, 2, 3, 4], y = td_0_05[0, 2:], color = (63/256, 184/256, 175/256) )

plt.plot(td_0_05[9, 2:], color = (127/256, 199/256, 175/256) )
plt.scatter(x = [0, 1, 2, 3, 4], y = td_0_05[9, 2:], color = (127/256, 199/256, 175/256) )

plt.plot(td_0_05[49, 2:], color = (218/256, 216/256, 167/256) )
plt.scatter(x = [0, 1, 2, 3, 4], y = td_0_05[49, 2:], color = (218/256, 216/256, 167/256) )

plt.plot(td_0_05[99, 2:], color = (255/256, 158/256, 157/256) )
plt.scatter(x = [0, 1, 2, 3, 4], y = td_0_05[-1, 2:], color = (255/256, 158/256, 157/256) )

plt.plot(true_values, '--', color = (154/256, 154/256, 154/256), linewidth = 2 , alpha = 0.5)
plt.scatter(x = [0, 1, 2, 3, 4], y = true_values, color = (154/256, 154/256, 154/256) )


plt.rcParams.update({'font.size': 15})
plt.legend(["1 episode", "10 episodes", "50 episodes", "100 episodes", "True values"])
plt.xlabel("States", fontsize = 23)
plt.ylabel("Value function", fontsize = 23)
plt.xticks([0, 1, 2, 3, 4], ['A', 'B', 'C', 'D', 'E'])

plt.show()








