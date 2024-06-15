import matplotlib.pyplot as plt
import numpy as np
import time
import math
import config
from PIL import Image

# ---- Constants ----

alpha = config.alpha
size = config.gridsize
impact_range = config.impact_range
iterations = config.iterations

# Generate grid

opinions = np.random.choice([-1, 1], size=(size, size))
stability = np.random.random((size, size)) # stability of opinion for each individual
spreading = np.random.random((size, size)) # readiness for each individual to spread their opinion

distance_weighting = np.empty(shape=(impact_range*2+1,impact_range*2+1)) # precalculate distance weighting factor for impact
for x in range(impact_range*2+1):
  for y in range(impact_range*2+1):
    if x != impact_range or y != impact_range:
      distance_weighting[x][y] = 1/((abs(x-impact_range)+abs(y-impact_range))**alpha)

def update_opinions():
  random_numbers = np.random.random((size, size))
  opinions_new = []
  for i in range(size):
    append = []
    for j in range(size):
      append.append(0)
    opinions_new.append(append) # initialize new opinion grid
  
  for x in range(size):
    for y in range(size):
      total_impact = 0
      for i in range(-impact_range,impact_range+1):
        for j in range(-impact_range,impact_range+1):
          if i != 0 or j != 0:
            origin_x = np.mod(x+i,size)
            origin_y = np.mod(y+j,size)
            total_impact += distance_weighting[i+impact_range][j+impact_range] * opinions[origin_x][origin_y] * spreading[origin_x][origin_y]
      raw_prob = 1/(1+math.exp(-total_impact)) #logistic function for impact -> probability
      prob = raw_prob * (1-stability[x][y])
      if opinions[x][y] == 1:
        prob += stability[x][y]
      if random_numbers[x][y] < prob:
        opinions_new[x][y] = 1
      else:
        opinions_new[x][y] = -1
  return opinions_new

start_time = time.time()
for i in range(iterations):
  opinions = update_opinions()
  color_grid = np.zeros((size, size, 3), dtype=np.uint8)
  for x in range(size):
    for y in range(size):
      if opinions[x][y] > 0:
        color_grid[x][y] = [255,255,255]

  image = Image.fromarray(color_grid)
  image.save('Output/output{0}.jpg'.format(i))
end_time=time.time()
print('Simulation finished. {0} seconds used'.format(end_time-start_time))
input("Press Enter to exit...")
