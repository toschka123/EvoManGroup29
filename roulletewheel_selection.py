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
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

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
                enemies=[6],
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
mutation_strength = 0.1


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

# def mutate(individual):
#     mutation_strength = 0.4  # You can adjust this value based on your problem

#     for i in range(len(individual)):
#         if random.random() < mutation_strength:
#             individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range

#     return individual

# def mutate_gene_gaussian(gene):
#     mutation = np.random.normal(0, 0.5)

#     if mutation + gene > 1: #if values too big
#         return 0.99
#     elif mutation < -1: #if values too small
#         return -0.99

#     gene += mutation

#     return gene

def mutate_gene_gaussian(gene, mutation_strength=0.65, min_value=-1, max_value=1):
    mutation = np.random.normal(0, mutation_strength)

    mutated_gene = gene + mutation

    # Clip the mutated gene to stay within the specified range
    mutated_gene = np.clip(mutated_gene, min_value, max_value)

    return mutated_gene


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

def roulette_wheel_selection(population, fitness_values, num_parents):
    total_fitness = sum(fitness_values)
    selected_parents = []

    for _ in range(num_parents):
        # Generate a random number between 0 and the total fitness
        rand_num = random.uniform(0, total_fitness)
        
        # Initialize variables for selecting an individual
        cumulative_fitness = 0
        selected_index = -1
        
        # Choose an individual based on roulette wheel selection
        for i in range(len(population)):
            cumulative_fitness += fitness_values[i]
            if cumulative_fitness >= rand_num:
                selected_index = i
                break
        
        if selected_index != -1:
            selected_parents.append(population[selected_index])
    
    return selected_parents

fitness_history = []  # Store fitness values for each generation


while Gen < maxGens:
    parents = []
    for i in range(int(N_newGen/2)):  # Choose how many pairs of parents we want
        parents.append(parent_selection(pop, pop_f, 4, N_newGen))
    new_kids = []
    for pairs in parents:  # Each pair of parents generates two offspring
        baby1, baby2 = recombination(pairs[0], pairs[1])
        new_kids.append(baby1)
        new_kids.append(baby2)

    # Roulette Wheel Selection for survivor selection
    selected_survivors = roulette_wheel_selection(pop, pop_f, N_newGen)

    # Replace old individuals with new offspring
    for i in range(len(selected_survivors)):
        pop[i] = selected_survivors[i]

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



