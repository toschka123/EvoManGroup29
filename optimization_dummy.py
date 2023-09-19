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
headless = True
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
                speed="fastest",
                visuals=False)


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# start writing your own code from here

pop_size = 100
max_f =-1
avg_f =-1
low_f = 999
maxGens=10
Gen=0
N_newGen=100 # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!

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

"""def parent_selection(population, f_vals): #random 50, change to something sensible later
    list_of_parents=[]
    for i in range(20):
        ip1 = randint(0,len(population)-1)
        ip2 = randint(0,len(population)-1)
        p1 = population[ip1]
        p2 = population[ip2]
        list_of_parents.append([p1, p2])
    return list_of_parents"""

def parent_selection(population, f_values, tournament_size=4, N_newGen = N_newGen/2): #Generate Pairs of parents and return them
    num_parents = N_newGen
    selected_parents = []

    for _ in range(num_parents):
        tournament_indices = np.random.choice(num_parents, size=tournament_size, replace=False)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [f_values[i] for i in tournament_indices]

        # Choose the best individual from the tournament as the parent
        best_index = np.argmax(tournament_fitness)
        selected_parents.append(tournament_individuals[best_index])
    return selected_parents

def kill_people(population, howManyShouldDie): #kill random individual
    choiceInd = random.sample(range(0,len(population)), howManyShouldDie)
    return choiceInd


"""def kill_tournament(population, f_values, tournament_size=8): 
    num_parents = len(population)
    selected_deaths = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(num_parents, size=tournament_size, replace=False)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [f_values[i] for i in tournament_indices]
        # Choose the best individual from the tournament as the parent
        best_index = np.argmin(tournament_fitness)
        selected_deaths.append(tournament_individuals[best_index])
    return selected_deaths"""

def mutate(individual):
    mutation_strength = 0.1  # You can adjust this value based on your problem

    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range

    return individual



while Gen < maxGens:
    parents=[]
    for i in range(int(N_newGen/2)): #Choose how many pairs of parents we want
        parents.append(parent_selection(pop, pop_f,4, N_newGen))
    new_kids = []
    for pairs in parents: #Each pair of parents generates two offspring
        baby1, baby2 = recombination(pairs[0], pairs[1])
        new_kids.append(baby1)
        new_kids.append(baby2)
    inds = kill_people(pop, N_newGen) # pick how many individuals we want to kill
    for i in range(len(inds)): #execute them
        personToDie=inds[i]
        pop[personToDie] = new_kids[i]
    Gen+=1
    pop_f = evaluate(env,pop) #evaluate new population
    max_f = max(pop_f)
    avg_f = sum(pop_f) / len(pop_f)
    low_f = min(pop_f)
    print(max_f, avg_f,len(pop))




