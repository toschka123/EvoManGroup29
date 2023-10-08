
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                enemies=[6],
=======
                enemies=[1, 4, 7],
                multiplemode="yes",
>>>>>>> Stashed changes
=======
                enemies=[5],
>>>>>>> Stashed changes
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
max_f =1
avg_f =-1
low_f = 999
<<<<<<< Updated upstream
maxGens=30
Gen=0
N_newGen=pop_size # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
fitness_survivor_no = 20 
mutation_strength = 0.1
gaussian_mutation_sd = 0.5 

pop = np.random.uniform(-1, 1, (pop_size, n_vars)) #initialize population
pop_f = evaluate(env,pop) #evaluate population
max_f=max(pop_f)
=======
maxGens = 20
Gen = 0
N_newGen = pop_size * 4  # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
mutation_threshold = 0.1
fitness_survivor_no = 20  # how many children in the new generation will be from "best". The rest are random.
gaussian_mutation_sd = 0.5
overall_best = -1
e0 = 0.02               #Formulate the boundary condition for sigma'
#COMPLETELY RANDOM NR NOW !!

fitness_avg_history = []
fitness_best_history = []
fitness_history = []

#Generate 266 genes, at loc 0 we find the sigma, the rest of the array is the weights
pop = np.random.uniform(-1, 1, (pop_size, n_vars)) #Initialize population, with extra value for the weights

#Define the bounds of your initial sigma values and tao for self-adaptive mutation
sigma_i_U = 0.1
sigma_i_L = 0.01

#Generate the stepsize (mutation size) of your sigma value
tao = 0.2
step_size = math.e ** (tao * np.random.normal(0, 1))

#Decide which baby to mutate


#Generate the initial sigma values and place them at location 0 for each individual array
sigma_vals_i = [random.uniform(sigma_i_U,sigma_i_L) for individual in range(pop_size)]
pop[:, 0] = sigma_vals_i
avg_sigma_start = sum(sigma_vals_i)/len(sigma_vals_i)

#Evaluate population
pop_f = evaluate(env,pop_weights_only(pop))
max_f = max(pop_f)
>>>>>>> Stashed changes
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

def mutate_gene_gaussian(gene):
    mutation = np.random.normal(0, 0.5)

    if mutation + gene > 1: #if values too big
        return 0.99
    elif mutation < -1: #if values too small
        return -0.99

    gene += mutation

    return gene


<<<<<<< Updated upstream
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range
    return individual


def adaptive_tournament_selection(population, f_values, min_tournament_size=4, max_tournament_size=10):
    num_parents = len(population)
    selected_parents = []  # List to store the selected parents
=======
# Tournament that decides which parents should create the new generation
def adaptive_tournament_selection(population, f_values, min_tournament_size=4, max_tournament_size=8):
    num_parents = int(len(population)/2)
    selected_parents = []
>>>>>>> Stashed changes

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


def assign_age(population):
    # Assign random ages to individuals in the population
    ages = [random.randint(0, 10) for _ in range(len(population))]
    return ages

def age_based_survivor_selection(population, fitness_values, pop_size, max_age=10, tournament_size=6):
    num_individuals = len(population)
    selected_survivors = []

    # Combine population, fitness values, and ages for sorting
    combined_data = list(zip(population, fitness_values, assign_age(population)))

    # Sort by fitness values in descending order
    combined_data.sort(key=lambda x: x[1], reverse=True)

    # Extract the sorted population, fitness values, and ages
    sorted_population, sorted_fitness_values, sorted_ages = zip(*combined_data)

    # Perform tournament selection for the remaining individuals
    num_tournament_rounds = pop_size
    for _ in range(num_tournament_rounds):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(num_individuals), tournament_size)

        # Find the best individual in the tournament based on fitness
        best_individual = max(tournament_indices, key=lambda x: sorted_fitness_values[x])

        # Get the age of the best individual
        best_individual_age = sorted_ages[best_individual]

        # Increment the age of the selected individual
        selected_individual_age = best_individual_age + 1

<<<<<<< Updated upstream
        # If the selected individual's age exceeds the maximum age, replace it
        if selected_individual_age <= max_age:
            selected_survivors.append(sorted_population[best_individual])
=======
elif run_mode == 'train':
  for run_number in range(1): #define how many times to run the experiment
    #Reinitialize parameters for each of the test runs
    max_f = -1
    avg_f = -1
    low_f = 999
>>>>>>> Stashed changes

    # Combine selected survivors to form the final population
    final_population = list(selected_survivors)

    return final_population

def mutate(individual):
    mutation_strength = 0.1  # You can adjust this value based on your problem

    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range

    return individual

max_age = 15  # Maximum age for individuals

# Initialize ages for the initial population
ages = assign_age(pop)

# Store fitness history for each generation
fitness_history = []

while Gen < maxGens:
    parents = []
    for i in range(int(N_newGen/2)):
        parents.append(adaptive_tournament_selection(pop, pop_f, 6, N_newGen))

    new_kids = []
    for pairs in parents:
        baby1, baby2 = recombination(pairs[0], pairs[1])
        new_kids.append(baby1)
        new_kids.append(baby2)

    # Update ages for the current population
    ages = [min(age + 1, max_age) for age in ages]

    # Combine old and new generations
    old_generation = pop.tolist()
    total = old_generation + new_kids
    total = np.array(total)
    total_f = evaluate(env, total)

    # Select survivors using age-based survivor selection
    survivors = age_based_survivor_selection(total, total_f, pop_size, max_age, tournament_size=6)
    # Update the population with survivors and their ages
    for i in range(len(survivors)):
        pop[i] = survivors[i]
        ages[i] = 0  # Reset the age of survivors

    Gen += 1
    pop_f = evaluate(env, pop)  # Evaluate the new population
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