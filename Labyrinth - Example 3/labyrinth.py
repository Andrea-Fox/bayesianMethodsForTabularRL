import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math

import random



class GridWorld:
    terrain_color = dict(normal=[127/360, 0, 96/100],
                         objective = [26/360, 100/100, 100/100],
                         water = [247/360, 92/100, 70/100],
                         player = [344/360, 93/100, 100/100], 
                         wall = [0, 0, 0])
        
    def __init__(self):
        self.player = None
        self._create_grid()  
        self._draw_grid()
        
    def _create_grid(self, initial_grid=None):
        self.grid = self.terrain_color['normal'] * np.ones((10, 10, 3))
        self._add_objectives(self.grid)
        
    def _add_objectives(self, grid):
        grid[8, 3:9] = self.terrain_color['water']
        grid[5, 0:6] = self.terrain_color['wall']
        grid[2, 2:9] = self.terrain_color['wall']
        grid[3:8, 8] = self.terrain_color['wall']
        grid[0, 9] = self.terrain_color['objective']
        
    def _draw_grid(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.grid(which='minor')       
        self.q_texts = [self.ax.text(*self._id_to_position(i)[::-1], '0',
                                     fontsize=11, verticalalignment='center', 
                                     horizontalalignment='center') for i in range(10 * 10)]     
         
        self.im = self.ax.imshow(hsv_to_rgb(self.grid), cmap='terrain',
                                 interpolation='nearest', vmin=0, vmax=1)        
        self.ax.set_xticks(np.arange(10))
        self.ax.set_xticks(np.arange(10) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(10))
        self.ax.set_yticks(np.arange(10) - 0.5, minor=True)
        
    def reset(self):
        self.player = (9,0)        
        return self._position_to_id(self.player)
    
    def step(self, action):
        # Possible actions
        if action == 0 and self.player[0] > 0 :
            if any( self.grid[self.player[0] - 1, self.player[1]] != self.terrain_color['wall']) : # go down
                self.player = (self.player[0] - 1, self.player[1])
        if action == 1 and self.player[0] < 9 :
            if any(self.grid[self.player[0] + 1, self.player[1]] != self.terrain_color['wall']): #  go up
                self.player = (self.player[0] + 1, self.player[1])
        if action == 2 and self.player[1] < 9 :
            if any(self.grid[self.player[0], self.player[1] + 1] != self.terrain_color['wall']): # go right
                self.player = (self.player[0], self.player[1] + 1)
        if action == 3 and self.player[1] > 0 :
            if any(self.grid[self.player[0], self.player[1] - 1] != self.terrain_color['wall']): # go left
                self.player = (self.player[0], self.player[1] - 1)
            
        # Rules
        if all(self.grid[self.player] == self.terrain_color['water']):
            reward = -100
            done = True
        elif all(self.grid[self.player] == self.terrain_color['objective']):
            reward = 0
            done = True
        else:
            reward = -1
            done = False
            
        return self._position_to_id(self.player), reward, done
    
    def _position_to_id(self, pos):
        return pos[0] * 10 + pos[1]
    
    def _id_to_position(self, idx):
        return (idx // 10), (idx % 10)
        
    def render(self, q_values=None, action=None, max_q=False, colorize_q=False):
        assert self.player is not None, 'You first need to call .reset()'  
        
        if colorize_q:
            assert q_values is not None, 'q_values must not be None for using colorize_q'            
            grid = self.terrain_color['normal'] * np.ones((10, 10, 3))
            values = change_range(np.max(q_values, -1)).reshape(10, 10)
            grid[:, :, 1] = values
            self._add_objectives(grid)
        else:            
            grid = self.grid.copy()
            
        grid[self.player] = self.terrain_color['player']       
        self.im.set_data(hsv_to_rgb(grid))
               
        if q_values is not None:
            xs = np.repeat(np.arange(10), 10)
            ys = np.tile(np.arange(10), 10)  
            
            for i, text in enumerate(self.q_texts):
                if max_q:
                    q = max(q_values[i])    
                    txt = '{:.2f}'.format(q)
                    text.set_text(txt)
                else:                
                    actions = ['U', 'D', 'R', 'L']
                    txt = '\n'.join(['{}: {:.2f}'.format(k, q) for k, q in zip(actions, q_values[i])])
                    text.set_text(txt)
                
        if action is not None:
            self.ax.set_title(action, color='r', weight='bold', fontsize=32)

        plt.pause(0.01)