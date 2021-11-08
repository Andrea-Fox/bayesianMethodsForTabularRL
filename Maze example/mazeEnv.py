import numpy as np

class Maze():
    terrain_color = dict(normal=[127/360, 0, 96/100],
                         objective = [26/360, 100/100, 100/100],
                         flag_1 = [247/360, 92/100, 70/100],
                         flag_2 = [248/360, 92/100, 70/100],
                         flag_3 = [249/360, 92/100, 70/100],
                         wall = [0, 0, 0],
                         starting_place = [344/360, 93/100, 100/100])

    def __init__(self):
        self.player = None
        self._create_grid() 
        self.flags_visited = [False, False, False]
        self.max_steps_episode = 100
        self.num_states = 49*8
        # self._draw_grid()

    
        
    def _create_grid(self, initial_grid=None):
        self.grid = self.terrain_color['normal'] * np.ones((7, 7, 3))
        self._add_objectives(self.grid)
        
    def _add_objectives(self, grid):
        grid[0,0] = self.terrain_color['starting_place']
        grid[3, 0:2] = self.terrain_color['wall']
        grid[0:2, 1] = self.terrain_color['wall']
        grid[0:2, 4] = self.terrain_color['wall']
        grid[3, 5:7] = self.terrain_color['wall']
        grid[6, :] = self.terrain_color['wall']

        grid[5, 0] = self.terrain_color['flag_1']
        grid[0, 2] = self.terrain_color['flag_2']
        grid[4, 6] = self.terrain_color['flag_3']
        
        grid[0, -1] = self.terrain_color['objective']

        
    def reset(self):
        self.player = (0, 0) 
        self.flags_visited = [False, False, False]
        return self._position_to_id(self.player)
    
    def step(self, action):
        # Possible actions
        slip_probability = 0.1
        random_number = np.random.random(1)[0]
        if random_number < slip_probability:
            action = (action+1)%4


        if action == 0 and self.player[0] > 0 :
            if any( self.grid[self.player[0] - 1, self.player[1]] != self.terrain_color['wall']) : # go down
                self.player = (self.player[0] - 1, self.player[1])
        if action == 1 and self.player[0] < 6 :
            if any(self.grid[self.player[0] + 1, self.player[1]] != self.terrain_color['wall']): #  go up
                self.player = (self.player[0] + 1, self.player[1])
        if action == 2 and self.player[1] < 6 :
            if any(self.grid[self.player[0], self.player[1] + 1] != self.terrain_color['wall']): # go right
                self.player = (self.player[0], self.player[1] + 1)
        if action == 3 and self.player[1] > 0 :
            if any(self.grid[self.player[0], self.player[1] - 1] != self.terrain_color['wall']): # go left
                self.player = (self.player[0], self.player[1] - 1)
            
        # Rules
        if all(self.grid[self.player] == self.terrain_color['flag_1']) and not self.flags_visited[0]:
            reward = -1
            self.flags_visited[0] = True
            done = False
        elif all(self.grid[self.player] == self.terrain_color['flag_2']) and not self.flags_visited[1]:
            reward = -1
            self.flags_visited[1] = True
            done = False
        elif all(self.grid[self.player] == self.terrain_color['flag_3']) and not self.flags_visited[2]:
            reward = -1
            self.flags_visited[2] = True
            done = False
        elif all(self.grid[self.player] == self.terrain_color['objective']):
            reward = 50 * sum(self.flags_visited)
            done = True
        else:
            reward = -1
            done = False
        
        return self._position_to_id(self.player), reward, done
    
    def _position_to_id(self, pos):
        return (pos[0] * 7 + pos[1])*8 + self.flags_visited[0] + 2 * self.flags_visited[1] + 4*self.flags_visited[2]