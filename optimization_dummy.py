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

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

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
tournament_size = 4

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

# def parent_selection(population,f_value): #random 50, change to something sensible later
#     list_of_parents=[]
#     for i in range(25):
#         ip1 = randint(0,len(population)-1)
#         ip2 = randint(0,len(population)-1)
#         p1 = population[ip1]
#         p2 = population[ip2]
#         list_of_parents.append([p1, p2])
#     return list_of_parents

<<<<<<< Updated upstream
def parent_selection(population, f_values, tournament_size):
=======
# def parent_selection(population, f_values, tournament_size=4):
#     num_parents = len(population)
#     selected_parents = []

#     for _ in range(num_parents):
#         tournament_indices = np.random.choice(num_parents, size=tournament_size, replace=False)
#         tournament_individuals = [population[i] for i in tournament_indices]
#         tournament_fitness = [f_values[i] for i in tournament_indices]

#         # Choose the best individual from the tournament as the parent
#         best_index = np.argmax(tournament_fitness)
#         selected_parents.append(tournament_individuals[best_index])

#     return selected_parents
    
def parent_selection(population, f_values, initial_tournament_size=4, max_tournament_size=8):
>>>>>>> Stashed changes
    num_parents = len(population)
    selected_parents = []

    # Track the diversity of individuals
    diversity_scores = np.zeros(num_parents)

    # Adaptive tournament size parameters
    current_tournament_size = initial_tournament_size
    tournament_size_increment = 1  # Adjust as needed

    for _ in range(num_parents):
        # Randomly select individuals for the tournament
        tournament_indices = np.random.choice(num_parents, size=current_tournament_size, replace=False)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [f_values[i] for i in tournament_indices]

        # Calculate the diversity score for each selected individual
        for i, index in enumerate(tournament_indices):
            diversity_scores[index] += np.mean(np.abs(tournament_fitness - f_values[index]))

        # Choose the best individual from the tournament as the parent
        best_index = tournament_indices[np.argmax(tournament_fitness)]

        # Update tournament size for the next selection (adaptive)
        if current_tournament_size < max_tournament_size:
            current_tournament_size += tournament_size_increment

        selected_parents.append(population[best_index])

    return selected_parents

def mutate(individual):
    mutation_strength = 0.1  # You can adjust this value based on your problem
    
    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range
    
    return individual

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



