import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe_values = pd.read_csv("data/dataframe_values_20_states.csv")


dataframe_values = dataframe_values.iloc[:, 1:]

print(dataframe_values)

unique_alpha_values = len(set(dataframe_values.iloc[:, 1]))
print(unique_alpha_values)
unique_n_values = set(dataframe_values.iloc[:, 0])
len_unique_n_values = len(unique_n_values)
print(len_unique_n_values)
list_unique_n_values = list(unique_n_values)
list_unique_n_values = sorted(list_unique_n_values)
print(list_unique_n_values)


fig = plt.figure(figsize=(10,6))
for n in range(1,len_unique_n_values+1): 
    plt.plot(dataframe_values.iloc[(n-1)*unique_alpha_values:n*unique_alpha_values, 1], dataframe_values.iloc[(n-1)*unique_alpha_values:n*unique_alpha_values, -1])


plt.rcParams.update({'font.size': 15})
plt.ylabel("RMSE", fontsize = 15)
plt.xlabel(r"$\alpha$", fontsize = 15)
plt.xlim((-0.05, 0.5))
plt.legend([str(n) + " steps" for n in list_unique_n_values], ncol = 2)
plt.show()





