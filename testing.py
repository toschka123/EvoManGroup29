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

headless = False
n_hidden_neurons = 10

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'optimization_test'

env = Environment(experiment_name=experiment_name,
                enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


def individual_gain(env, individual):
    f,p,e,t = env.play(pcont=individual)
    indiv_gain = int(p)-int(e)
    return indiv_gain

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def evaluate_indiv(bsol):
    return(individual_gain(env, bsol))

bsol = np.loadtxt(experiment_name + '/best.txt')
print(evaluate_indiv(bsol))

sys.exit(0)