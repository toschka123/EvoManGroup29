###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import math
from evoman.environment import Environment
from demo_controller import player_controller
from random import randint, random
from save_run import save_run
import random
# imports other libs
import numpy as np
import os
import matplotlib.pyplot as plt 


# Runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# Evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


# Returns the population with only the 265 weights, no sigma
def pop_weights_only(pop):
    weights_only = pop[:,1:]
    return weights_only


# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[7],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),  # You  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)


# Number of variables for multilayer with 10 hidden neurons (265) plus one sigma value
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 +1

# Start writing your own code from here

# Initialization
run_mode = "train"
pop_size = 100
max_f = -1
avg_f = -1
low_f = 999
maxGens = 20
Gen = 0
N_newGen = pop_size * 4  # Define number of offspring generated and old individuals to kill (This has to be even!!)
mutation_threshold = 0.04  # Increasing lowers the chance of mutation
fitness_survivor_no = 20  # How many children in the new generation will be from "best". The rest are random.
gaussian_mutation_sd = 0.5
overall_best = -1
fitness_avg_history = []
fitness_best_history = []
fitness_history = []

# Initialise the population, with each individual having 265 genes(weights) and one extra for the sigma
pop = np.random.uniform(-1, 1, (pop_size, n_vars))

# Define bounds for the initial sigma values and tau for self-adaptive mutation
sigma_i_U = 0.1
sigma_i_L = 0.01
tau = 0.05

# Generate the initial sigma values and place them at index 0 for each individual
sigma_vals_i = [random.uniform(sigma_i_U,sigma_i_L) for individual in range(pop_size)]
pop[:, 0] = sigma_vals_i
avg_sigma_start = sum(sigma_vals_i)/len(sigma_vals_i)

# Evaluate initial population
pop_f = evaluate(env,pop_weights_only(pop))


# Takes two parents and returns 2 babies with each gene having a 50% chance to be from either parent
def uniform_recombination(i1, i2):
    baby1 = []
    baby2 = []

    # Start with choosing which sigma to inherit
    if randint(0, 1) == 1:
        baby1.append(i1[0])
        baby2.append(i2[0])
    else:
        baby2.append(i1[0])
        baby1.append(i2[0])

    # Decide if a baby will mutate
    if random.random() > mutation_threshold:

        # Generate the mutation size for the sigma value
        mutation_size = math.e ** (tau * np.random.normal(0, 1))

        # Decide which baby to mutate, multiplying its old sigma with the mutation size
        if randint(0, 1) == 1:
            sigma_prime = baby1[0] * mutation_size
            baby1[0] = sigma_prime

        else:
            sigma_prime = baby2[0] * mutation_size
            baby2[0] = sigma_prime

    sigma1 = baby1[0]
    sigma2 = baby2[0]

    for i in range(1, len(i1)):
        if randint(0, 1) == 1:
            baby1.append(i1[i])
            baby2.append(i2[i])
        else:
            baby1.append(i2[i])
            baby2.append(i1[i])

        if random.random() > mutation_threshold:
            baby1[i] = mutate_gene_sa(baby1[i], sigma1)
        if random.random() > mutation_threshold:
            baby2[i] = mutate_gene_sa(baby1[i], sigma2)

    return baby1, baby2


# Mutates a single gene
def mutate_gene_sa(gene, s):
    mutation = s * np.random.normal(0, 1)
    while abs(gene + mutation) > 1:  # Ensures mutations stay between (-1, 1)
        mutation = s * np.random.normal(0, 1)
    gene += mutation
    return gene


# Alternative way of mutation (not in use)
def mutate_gene_gaussian(gene):
    mutation = np.random.normal(0, gaussian_mutation_sd)

    while (mutation + gene > 1) or (mutation + gene < -1):
        mutation = np.random.normal(0, gaussian_mutation_sd)

    gene += mutation
    return gene


# Tournament that decides which parents should create the new generation
def adaptive_tournament_selection(population, f_values, min_tournament_size=2, max_tournament_size=5):
    num_parents = int(len(population)/2)
    selected_parents = []

    # Track the diversity of individuals using an array of zeros
    diversity_scores = np.zeros(num_parents)

    # Adaptive tournament size parameters
    current_tournament_size = min_tournament_size  # Initialize the tournament size
    tournament_size_increment = 1  # Increment to adjust tournament size (can be modified)

    # Loop over the number of parents to select
    for _ in range(int(N_newGen)):
        # Randomly select individuals for the tournament (without replacement)
        tournament_indices = np.random.choice(num_parents, size=min_tournament_size, replace=False)
        # Calculate the fitness values of the selected individuals
        tournament_fitness = [f_values[i] for i in tournament_indices]
        best_index1 = tournament_indices[np.argmax(tournament_fitness)]

        # Choose the best individual from the tournament as the parent
        tournament_indices = np.random.choice(num_parents, size=max_tournament_size, replace=False)
        # Calculate the fitness values of the selected individuals
        tournament_fitness = [f_values[i] for i in tournament_indices]
        best_index2 = tournament_indices[np.argmax(tournament_fitness)]

        # Append the best individual to the list of selected parents
        selected_parents.append(population[best_index1])
        selected_parents.append(population[best_index2])

    return selected_parents  # Return the list of selected parents


# Decides which children survive and returns them in an array
def survivor_selector_mu_lambda(children, no_best_picks):

    survivors = np.random.uniform(-1, 1, (pop_size, n_vars))  # Preallocate a random array for survivors

    children_without_sigma = pop_weights_only(children)
    children_fitness = evaluate(env, children_without_sigma)
    # Ranks the children based on their fitness value
    indices_best_children = np.argpartition(children_fitness, -no_best_picks)[-no_best_picks:]

    for i in range(no_best_picks):  # Adds the best children to the new population
        survivors[i] = children[indices_best_children[i]]
    
    for i in range(no_best_picks, pop_size):  # Fills remaining population with random children
        survivors[i] = children[random.randint(0, pop_size-1)]

    return survivors


if run_mode == 'test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    env.update_parameter('visuals', True)
    evaluate(env, [bsol])

    sys.exit(0)

elif run_mode == 'train':

    for run_number in range(10):
        # Resets the initial parameters
        pop_size = 100
        maxGens = 20
        Gen = 0
        N_newGen = pop_size * 4
        mutation_threshold = 0.04
        fitness_survivor_no = 20
        gaussian_mutation_sd = 0.5
        overall_best = -1
        fitness_avg_history = []
        fitness_best_history = []
        fitness_history = []

        pop = np.random.uniform(-1, 1, (pop_size, n_vars))

        sigma_i_U = 0.1
        sigma_i_L = 0.01
        tau = 0.05

        sigma_vals_i = [random.uniform(sigma_i_U, sigma_i_L) for individual in range(pop_size)]
        pop[:, 0] = sigma_vals_i
        avg_sigma_start = sum(sigma_vals_i) / len(sigma_vals_i)

        # Evaluate population
        pop_f = evaluate(env, pop_weights_only(pop))
        max_f = max(pop_f)
        avg_f = sum(pop_f) / len(pop_f)
        low_f = min(pop_f)
        print(max_f, avg_f)

        # Run once for each generation
        while Gen < maxGens:

            parents = adaptive_tournament_selection(pop, pop_f, 4)  # Generates parents
            new_kids = np.random.uniform(-1, 1, (N_newGen, n_vars))  # Preallocate kids

            # Generate the new kids
            for i in range(0, N_newGen, 2):
                baby1, baby2 = uniform_recombination(parents[i], parents[i+1])
                new_kids[i] = baby1
                new_kids[i + 1] = baby2

            # Decide which kids survive
            survivors = survivor_selector_mu_lambda(new_kids, fitness_survivor_no)

            # Overwrite the old population with the new survivors generation
            for i in range(pop_size):
                pop[i] = survivors[i]

            Gen += 1

            pop_without_sigma = pop_weights_only(pop)

            # evaluate new population
            pop_f = evaluate(env, pop_without_sigma)
            max_f = max(pop_f)
            avg_f = sum(pop_f) / len(pop_f)
            low_f = min(pop_f)
            print(max_f, avg_f)

            fitness_avg_history.append((avg_f))

            # Saves the best individual
            if max_f > overall_best:
                overall_best = max_f
                best = np.argmax(pop_f)
                best_individual = pop_without_sigma[best]

                np.savetxt(experiment_name + '/best.txt', pop_without_sigma[best])

            fitness_best_history.append(overall_best)
            # Calculate and store the fitness values of the current population
            fitness_values = evaluate(env, pop_without_sigma)
            fitness_history.append(fitness_values)
            # Calculate the standard deviation of fitness values
            fitness_std = np.std(fitness_values)

            # Print the fitness diversity metric for the current generation
            print(f"Generation {Gen}: Fitness Diversity (Std Dev): {fitness_std}")

        avg_sigma_end = sum(pop[:,0])/len(pop[:,0])
        save_run(fitness_avg_history, fitness_best_history, avg_sigma_start, avg_sigma_end, "waterman", run_number)

        # Optionally visualize the fitness diversity over generations
        plt.plot(range(maxGens), [np.std(fitness) for fitness in fitness_history])
        plt.title("Fitness Diversity Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Standard Deviation of Fitness")
        plt.show()


