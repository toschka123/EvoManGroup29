
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
import matplotlib.pyplot as plt 

from scipy.spatial import distance
from scipy.cluster import hierarchy
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
maxGens=20
Gen=0
N_newGen=pop_size # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
mutation_strength = 0.04
fitness_survivor_no = 20 # how many children in the new generation will be from "best". The rest are random.
gaussian_mutation_sd = 0.5 

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

        if random.random() > mutation_strength:
            baby1[i] = mutate_gene_gaussian(baby1[i])

    return baby1, baby2


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range
    return individual


def mutate_gene_gaussian(gene):
    mutation = np.random.normal(0, 0.5)

    if mutation + gene > 1: #if values too big
        return 0.99
    elif mutation < -1: #if values too small
        return -0.99

    gene += mutation

    return gene


# def parent_selection(population, f_values, tournament_size):
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


def parent_selection(population, f_values, tournament_size=6, N_newGen = N_newGen/2): #Generate Pairs of parents and return them
    num_parents = N_newGen
    selected_parents = []

    #for _ in range(num_parents):
    tournament_indices = np.random.choice(num_parents, size=tournament_size, replace=False)
    tournament_individuals = [population[i] for i in tournament_indices]
    tournament_fitness = [f_values[i] for i in tournament_indices]

    # Choose the best individual from the tournament as the parent
    best_index = np.argmax(tournament_fitness)
    # Select the second parent randomly
    second_parent_ind = random.randint(0,99)
    selected_parents.append(tournament_individuals[best_index])
    selected_parents.append(population[second_parent_ind])
    return selected_parents

def kill_people(population, howManyShouldDie): #kill random individual
    choiceInd = random.sample(range(0,len(population)), howManyShouldDie)
    return choiceInd

def select_surv(pop, f_pop, N_remove=N_newGen):
    indxs= sorted(range(len(f_pop)), key=lambda k: f_pop[k])[N_remove:]
    survivors = []
    for i in indxs:
        survivors.append(pop[i])
    return survivors


# Returns a survivor array containing surviving children (only!).
# Some (small) number of surviving children are picked based on fitness.
# The rest are picked randomly.
def survivor_selector_mu_lambda(children, no_best_picks):
    survivors = np.random.uniform(-1, 1, (pop_size, n_vars)) #preallocate a random array for survivors

    children_fitness = evaluate(env, children)
    indices_best_children = np.argpartition(children_fitness, -no_best_picks)[-no_best_picks:]

    for i in range(no_best_picks): #add some number of best children to the new population
        survivors[i] =children[indices_best_children[i]] 
    
    for i in range(no_best_picks, pop_size): #fill the rest of the population with random children
        survivors[i] = children[random.randint(0, pop_size-1)]

    return survivors


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


fitness_history = []  # Store fitness values for each generation

while Gen < maxGens:
    parents=[]
    for i in range(int(N_newGen/2)): #Choose how many pairs of parents we want
        parents.append(parent_selection(pop, pop_f,4, N_newGen))
    new_kids = []
    for pairs in parents: #Each pair of parents generates two offspring
        baby1, baby2 = recombination(pairs[0], pairs[1])
        new_kids.append(baby1)
        new_kids.append(baby2)

    # parents=[]
    # parents = parent_selection(pop, pop_f, 4) #generates 100 parents - parent selection seems to make the convergion faster

    # new_kids = np.random.uniform(-1, 1, (6*pop_size, n_vars)) #preallocate 600 kids
    # for i in range(0,len(new_kids),2):
    #     baby1, baby2 = recombination(parents[random.randint(0, 99)], parents[random.randint(0, 99)])
    #     new_kids[i] = baby1
    #     new_kids[i+1] = baby2

    old_generation = pop.tolist()
    total = old_generation + new_kids
    total =np.array(total)
    total_f = evaluate(env, total)
    survivors = select_surv(total, total_f, 100)
    for i in range(len(survivors)):
        pop[i] = survivors[i]

    # survivors = select_surv(new_kids, fitness_survivor_no)
    # for i in range(pop_size):
    #     pop[i] = survivors[i]

    Gen+=1
    pop_f = evaluate(env,pop) #evaluate new population
    max_f = max(pop_f)
    avg_f = sum(pop_f) / len(pop_f)
    low_f = min(pop_f)
    print(max_f, avg_f,len(pop))

    # Calculate and store the fitness values of the current population
    fitness_values = evaluate(env, pop)
    fitness_history.append(fitness_values)

    # Calculate the standard deviation of fitness values
    fitness_std = np.std(fitness_values)

    # Print or log the fitness diversity metric for the current generation
    print(f"Generation {Gen}: Fitness Diversity (Std Dev): {fitness_std}")

# After the loop, you can visualize the fitness diversity over generations if needed
plt.plot(range(maxGens), [np.std(fitness) for fitness in fitness_history])
plt.title("Fitness Diversity Over Generations")
plt.xlabel("Generation")
plt.ylabel("Standard Deviation of Fitness")
plt.show()


