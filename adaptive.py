
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
                enemies=[3],
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
maxGens=30
Gen=0
N_newGen=pop_size # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
fitness_survivor_no = 20 
mutation_strength = 0.1
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


# def recombination(parent1, parent2, mutation_strength):
#     # Ensure the parents have the same length
#     assert len(parent1) == len(parent2), "Parents must have the same length"

#     baby1 = [gene1 if random.random() < 0.5 else gene2 for gene1, gene2 in zip(parent1, parent2)]
#     baby2 = [gene1 if random.random() < 0.5 else gene2 for gene1, gene2 in zip(parent2, parent1)]

#     # Apply mutation to baby1
#     baby1 = [mutate_gene_gaussian(gene, mutation_strength) for gene in baby1]

#     return baby1, baby2

def mutate_gene_gaussian(gene):
    mutation = np.random.normal(0, 0.5)

    if mutation + gene > 1: #if values too big
        return 0.99
    elif mutation < -1: #if values too small
        return -0.99

    gene += mutation

    return gene


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range
    return individual


def adaptive_tournament_selection(population, f_values, min_tournament_size=4, max_tournament_size=8):
    num_parents = len(population)
    selected_parents = []  # List to store the selected parents

    # Track the diversity of individuals using an array of zeros
    diversity_scores = np.zeros(num_parents)

    # Adaptive tournament size parameters
    current_tournament_size = min_tournament_size  # Initialize the tournament size
    tournament_size_increment = 1  # Increment to adjust tournament size (can be modified)

    # Loop over the number of parents to select
    for _ in range(num_parents):
        # Randomly select individuals for the tournament (without replacement)
        tournament_indices = np.random.choice(num_parents, size=current_tournament_size, replace=False)
        
        # Calculate the fitness values of the selected individuals
        tournament_fitness = [f_values[i] for i in tournament_indices]

        # Calculate the diversity score for each selected individual
    for index in tournament_indices:
        # Calculate the absolute differences between the fitness of the selected individual
        # and the fitness of other individuals in the tournament, then take the mean.
        diversity_scores[index] += np.mean(np.abs(tournament_fitness - f_values[index]))

        # Choose the best individual from the tournament as the parent
        best_index = tournament_indices[np.argmax(tournament_fitness)]

        # Append the best individual to the list of selected parents
        selected_parents.append(population[best_index])

        # Update tournament size for the next selection (adaptive)
        if current_tournament_size < max_tournament_size:
            current_tournament_size += tournament_size_increment

    return selected_parents  # Return the list of selected parents

def select_surv(pop, f_pop, N_remove=N_newGen):
    indxs= sorted(range(len(f_pop)), key=lambda k: f_pop[k])[N_remove:]
    survivors = []
    for i in indxs:
        survivors.append(pop[i])
    return survivors

def mutate(individual):
    mutation_strength = 0.1  # You can adjust this value based on your problem

    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range

    return individual


fitness_history = []  # Store fitness values for each generation

while Gen < maxGens:
    parents = []
    for i in range(int(N_newGen/2)):
        parents.append(adaptive_tournament_selection(pop, pop_f, 4, N_newGen))

    new_kids = []
    for pairs in parents:
        baby1, baby2 = recombination(pairs[0], pairs[1], mutation_strength)
        new_kids.append(baby1)
        new_kids.append(baby2)


    survivors = select_surv(new_kids, fitness_survivor_no)
    for i in range(pop_size):
        pop[i] = survivors[i]

    Gen += 1
    pop_f = evaluate(env, pop)  # evaluate new population
    max_f = max(pop_f)
    avg_f = sum(pop_f) / len(pop_f)
    low_f = min(pop_f)
    print(max_f, avg_f, len(pop))

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