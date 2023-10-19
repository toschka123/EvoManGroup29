import sys
import math
from evoman.environment import Environment
from demo_controller import player_controller
from random import randint, random
import random
# imports other libs
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

headless = True
n_hidden_neurons = 10

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'solutions_islanding_678'

env = Environment(experiment_name=experiment_name,
                enemies = [1],#enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                # multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

# test_env = Environment(experiment_name=experiment_name,
#                 enemies=[1, 2, 3, 4, 5, 6, 7, 8],
#                 multiplemode="yes",
#                 playermode="ai",
#                 player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
#                 enemymode="static",
#                 level=2,
#                 speed="fastest",
#                 visuals=False)

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def individual_gain(env, individual):
    f,p,e,t = env.play(pcont=individual)
    indiv_gain = float(p)-float(e)
    return indiv_gain, p, e

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

energyGain=[]
playerHealth=[]
enemyHealth = []

bsol = np.loadtxt(experiment_name + '/best_2.txt')
print('\n RUNNING SAVED BEST SOLUTION \n')

for i in range(1,9):
    env.update_parameter('enemies', [i])
    gain, p, e = individual_gain(env, bsol)
    playerHealth.append(p)
    enemyHealth.append(e)
    energyGain.append(gain)

print(sum(energyGain)/len(energyGain))
print(energyGain)
print(playerHealth)
print(enemyHealth)


table = [range(1,9), 
         playerHealth,
         enemyHealth]

print(tabulate(table))

sys.exit(0)