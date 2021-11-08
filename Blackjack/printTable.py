from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as mpatches

q_values = pd.read_csv("data/table_optimal_thorp.txt", sep = ",",  header=None)  
print(q_values)
q_values = q_values.iloc[:, 1:]
q_values = q_values.transpose()

print(q_values.shape)
print(q_values)

q_values.iloc[:, 0] = q_values.iloc[:, 0].astype(float)
q_values.iloc[:, 0] = q_values.iloc[:, 0].apply(np.floor)
q_values.iloc[:, :2] = q_values.iloc[:, :2].astype(int)


print(q_values.dtypes)


q_values_usable_ace = q_values.loc[q_values.iloc[:, 2] == 'True']
q_values_non_usable_ace = q_values.loc[q_values.iloc[:, 2] == 'False']
print(q_values_usable_ace)
print(q_values_non_usable_ace)
# q_values.to_csv("table_blackjack.txt")


# dobbiamo ora creare la griglia, in modo da mettere in evidenza quale è l'azione migliore a seconda del valore delle carte in mano e del dealer


tile_color = dict(  stick=[222/256, 41/256, 41/256],
                    hit = [41/256, 222/256, 71/256])


grid = tile_color['hit'] * np.ones((10, 10, 3))
grid[:, 0] = tile_color['stick']


# creiamo la grid corrispondente alla strategia quando l'asso è 'usable


# NON USABLE ACE

print(q_values_usable_ace[q_values_usable_ace.iloc[:, 0] == 12])


griglia_risultati = np.zeros((10, 10))

for player_value in range(12, 22):
    for dealer_value in range(1, 11):
        print("----------------------------")
        print(player_value, dealer_value)
        # se il valore massimo è quello nell'ultima colonna allora la mossa ottimale è prendere un'altra carta
        # salviamo nella posizione [player_index-11, dealer_index - 1]
        values_to_consider = q_values_non_usable_ace[(q_values_non_usable_ace.iloc[:, 0] == player_value) & (q_values_non_usable_ace.iloc[:, 1] == dealer_value)]
        #values_to_consider = q_values_usable_ace.iloc[q_values_usable_ace.iloc[:, 0] == player_index and q_values_usable_ace.iloc[:, 1] == dealer_index , -2:]
        print(float(values_to_consider.iloc[:, 3]))
        print(float(values_to_consider.iloc[:, 4]))
        # values_to_consider = values_to_consider.to_numpy()
        # print(values_to_consider)
        player_index = 21 - player_value
        dealer_index = dealer_value - 1


        if ( float(values_to_consider.iloc[:, 4]) > float(values_to_consider.iloc[:, 3]) ):
            print(player_value, dealer_value, "hit")
            grid[player_index, dealer_index] = tile_color['hit']
            griglia_risultati[player_index, dealer_index] = 1
        else:
            print(player_value, dealer_value, "stick")
            grid[player_index, dealer_index] = tile_color['stick']
            griglia_risultati[player_index, dealer_index] = 0


print("----------------------------")
print(griglia_risultati)



# code for showing the plot (as seen in https://stackoverflow.com/questions/38973868/adjusting-gridlines-and-ticks-in-matplotlib-imshow)

plt.figure()
im = plt.imshow(grid, interpolation='none', vmin=0, vmax=1, aspect='equal')

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

# Labels for major ticks
ax.set_xticklabels(np.arange(1, 11, 1))
ax.set_yticklabels(np.arange(21, 11, -1))

# Minor ticks
ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
ax.set_title("No usable ace", weight='bold', fontsize=18)

plt.xlabel('Dealer showing', fontsize = 14)
plt.ylabel('Player sum', fontsize = 14)

t = 0.75
cmap = {1:[222/256, 41/256, 41/256,t],2:[41/256, 222/256, 71/256,t]}
labels = {1:'stick',2:'hit'}    
## create patches as legend
patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]

plt.rcParams.update({'font.size': 13})    
plt.legend(handles=patches, loc=1, borderaxespad=0.)



plt.show()




# USABLE ACE

griglia_risultati = np.zeros((10, 10))

for player_value in range(12, 22):
    for dealer_value in range(1, 11):
        print("----------------------------")
        print(player_value, dealer_value)
        # se il valore massimo è quello nell'ultima colonna allora la mossa ottimale è prendere un'altra carta
        # salviamo nella posizione [player_index-11, dealer_index - 1]
        values_to_consider = q_values_usable_ace[(q_values_usable_ace.iloc[:, 0] == player_value) & (q_values_usable_ace.iloc[:, 1] == dealer_value)]
        #values_to_consider = q_values_usable_ace.iloc[q_values_usable_ace.iloc[:, 0] == player_index and q_values_usable_ace.iloc[:, 1] == dealer_index , -2:]
        print(float(values_to_consider.iloc[:, 3]))
        print(float(values_to_consider.iloc[:, 4]))
        # values_to_consider = values_to_consider.to_numpy()
        # print(values_to_consider)
        player_index = 21 - player_value
        dealer_index = dealer_value - 1


        if ( float(values_to_consider.iloc[:, 4]) > float(values_to_consider.iloc[:, 3]) ):
            print(player_value, dealer_value, "hit")
            grid[player_index, dealer_index] = tile_color['hit']
            griglia_risultati[player_index, dealer_index] = 1
        else:
            print(player_value, dealer_value, "stick")
            grid[player_index, dealer_index] = tile_color['stick']
            griglia_risultati[player_index, dealer_index] = 0


print("----------------------------")
print(griglia_risultati)

plt.figure()
im = plt.imshow(grid, interpolation='none', vmin=0, vmax=1, aspect='equal')

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

# Labels for major ticks
ax.set_xticklabels(np.arange(1, 11, 1))
ax.set_yticklabels(np.arange(21, 11, -1))

# Minor ticks
ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
ax.set_title("Usable ace", weight='bold', fontsize=18)

plt.xlabel('Dealer showing', fontsize = 14)
plt.ylabel('Player sum', fontsize = 14)



plt.legend(handles=patches, loc=1, borderaxespad=0.)

plt.show()



