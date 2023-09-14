###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller
from random import randint, random
import random
# imports other libs
import numpy as np
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))



# choose this for not using visuals and thus making experiments faster
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[2],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="normal",
                visuals=True)


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# start writing your own code from here

pop_size = 100
max_f =-1
avg_f =-1
low_f = 999
maxGens=10
Gen=0

pop = np.random.uniform(-1, 1, (pop_size, n_vars)) #initialize population
pop_f = evaluate(env,pop) #evaluate population
max_f=max(pop_f)
avg_f = sum(pop_f)/len(pop_f)
low_f = min(pop_f)
print(max_f, avg_f)

def recombination(i1, i2): #Takes as input two parents and returns 2 babies, in each position 50% chance to have parent1's gene
    baby1=[]
    baby2=[]
    for i in range(len(i1)):
        if randint(0,1) == 1:
            baby1.append(i1[i])
            baby2.append(i2[i])
        else:
            baby1.append(i2[i])
            baby2.append(i1[i])
    return baby1, baby2

def parent_selection(population,f_value): #random 50, change to something sensible later
    list_of_parents=[]
    for i in range(25):
        ip1 = randint(0,len(population)-1)
        ip2 = randint(0,len(population)-1)
        p1 = population[ip1]
        p2 = population[ip2]
        list_of_parents.append([p1, p2])
    return list_of_parents

def kill_people(population): #kill random individual
    for i in range(50):
        choiceInd = randint(0,len(population)-1)
        np.delete(population, choiceInd)
    return population


"""while Gen < maxGens:
    parents = parent_selection(pop, pop_f)
    new_kids = np.empty(1)
    for pairs in parents:
        baby1, baby2 = recombination(pairs[0], pairs[1])
        np.append(new_kids, baby1)
        np.append(new_kids,baby2)
    pop = kill_people(pop)
    new_kids= np.array(new_kids)
    pop = pop + new_kids

    pop_f = evaluate(env,pop)
    max_f = max(pop_f)
    avg_f = sum(pop_f) / len(pop_f)
    low_f = min(pop_f)
    print(max_f, avg_f,len(pop))
    Gen+=1"""



