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

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def evaluate_gain(env, individual):
    return np.array((map(lambda y: individual_gain(env, y),individual)))

def individual_gain(env, individual):
    f,p,e,t = env.play(pcont=individual)
    indiv_gain = int(p)-int(e)
    return indiv_gain

#Function that returns the population with only the 265 weights, no sigma
def pop_weights_only(pop):
    weights_only = pop[:,1:]
    return weights_only

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'solutions_islanding_678'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[6, 7, 8],
                multiplemode="yes",
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

test_env = Environment(experiment_name=experiment_name,
                enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                multiplemode="yes",
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

# number of variables for multilayer with 10 hidden neurons (265) plus one sigma value
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 +1

# start writing your own code from here
island_no = 2 # must be even
migrating_indivs_no = 10 # how many inidivduals migrate
migration_multiple = 5 # migration occurs every X gens
pop_size = int(100 / island_no)
max_f = -1
avg_f = -1
low_f = 999
maxGens = 20
Gen = 0
N_newGen = pop_size * 4  # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
mutation_threshold = 0.1
fitness_survivor_no = int(40 /island_no)  # how many children in the new generation will be from "best". The rest are random.
gaussian_mutation_sd = 0.5
overall_best = -1
e0 = 0.02               #Formulate the boundary condition for sigma'
run_mode = "train"

fitness_avg_history = []
fitness_best_history = []
fitness_history = []

#Define the bounds of your initial sigma values and tao for self-adaptive mutation
sigma_i_U = 0.5  # when starting at 1.0 scores dont really improve until it reaches around 0.5
sigma_i_L = 0.5  # 0.1 could be good too, but maybe converges too quick? 0.01 definitely too refined to start with

#Generate the stepsize (mutation size) of your sigma value
tao = 0.05
crossover_threshold = 12 

#Uniform recombination
def uniform_recombination(i1, i2): #Takes as input two parents and returns 2 babies, in each position 50% chance to have parent1's gene
    """
    Perform uniform recombination between two parents.

    Args:
    - i1 (list): The genetic representation of the first parent.
    - i2 (list): The genetic representation of the second parent.
    - mutation_threshold (float): Probability threshold for performing mutation.
    - step_size (float): Mutation step size for sigma adjustment.
    - e0 (float): Minimum value for the mutated sigma.

    Returns:
    - baby1 (list): The genetic representation of the first offspring.
    - baby2 (list): The genetic representation of the second offspring.
    """
    baby1=[]
    baby2=[]

    #Start with choosing which sigma to inherit
    if randint(0, 1) == 1:
        baby1.append(i1[0])
        baby2.append(i2[0])
    else:
        baby2.append(i1[0])
        baby1.append(i2[0])

    #Then possibly perform mutation on either the sigma of your first or your second baby
    if random.random() > mutation_threshold: #and assign it its new sigma
        if randint(0, 1) == 1:
            sigma_prime = baby1[0] * step_size
            if sigma_prime < e0:
                sigma_prime = e0
            baby1[0] = sigma_prime

        else:
            sigma_prime = baby2[0] * step_size
            if sigma_prime < e0:
                sigma_prime = e0
            baby2[0] = sigma_prime

    sigma1 = baby1[0]
    sigma2 = baby2[0]

    for i in range(1,len(i1)):
        if randint(0,1) == 1:
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


def mutate_gene_sa(gene, s):
    #Only perform mutation when result stays within weight range (-1,1)
    mutation = s * np.random.normal(0,1)
    while abs(gene + mutation) > 1:
        mutation = s * np.random.normal(0,1)
    gene += mutation
    return gene


def multipoint_crossover(parent1, parent2, mutation_threshold, tao, e0):
    """
    Perform multipoint crossover between two parents.

    Args:
    - parent1 (list): The first parent's genetic representation.
    - parent2 (list): The second parent's genetic representation.
    - mutation_threshold (float): Probability threshold for performing mutation.
    - tao (float): Mutation parameter for step size adjustment.
    - e0 (float): Minimum value for the mutated sigma.

    Returns:
    - baby1 (list): The genetic representation of the first offspring.
    - baby2 (list): The genetic representation of the second offspring.
    """
    # Check if parents have the same length
    assert len(parent1) == len(parent2), "Parents must have the same length."

    # Determine the number of crossover points randomly
    num_points = random.randint(1, len(parent1) - 1)
    
    # Generate sorted random crossover points
    crossover_points = sorted(random.sample(range(1, len(parent1)), num_points))

    # Initialize offspring containers
    baby1 = []
    baby2 = []

    # Iterate over crossover points to perform gene exchange
    for i in range(len(crossover_points) + 1):
        # Determine the start and end indices for gene exchange
        start = 0 if i == 0 else crossover_points[i - 1]
        end = len(parent1) if i == len(crossover_points) else crossover_points[i]

        # Alternate between parents to create offspring
        if i % 2 == 0:
            baby1.extend(parent1[start:end])
            baby2.extend(parent2[start:end])
        else:
            baby1.extend(parent2[start:end])
            baby2.extend(parent1[start:end])

    # Perform mutation on sigma values of baby1
    if random.random() > mutation_threshold:
        # Generate a random step size for mutation
        step_size = math.e ** (tao * random.normalvariate(0, 1))
        
        # Calculate the mutated sigma value
        sigma_prime1 = baby1[0] * step_size
        
        # Ensure the mutated sigma value does not go below e0
        if sigma_prime1 < e0:
            sigma_prime1 = e0
        
        # Update baby1's sigma value
        baby1[0] = sigma_prime1

    # Perform mutation on sigma values of baby2
    if random.random() > mutation_threshold:
        # Generate a random step size for mutation
        step_size = math.e ** (tao * random.normalvariate(0, 1))
        
        # Calculate the mutated sigma value
        sigma_prime2 = baby2[0] * step_size
        
        # Ensure the mutated sigma value does not go below e0
        if sigma_prime2 < e0:
            sigma_prime2 = e0
        
        # Update baby2's sigma value
        baby2[0] = sigma_prime2

    # Return the resulting offspring, baby1 and baby2
    return baby1, baby2

def adaptive_tournament_selection(population, f_values, min_tournament_size=2, max_tournament_size=5):
    """
    Perform adaptive tournament selection to choose parents from a population.

    Args:
    - population (list): List of individuals in the population.
    - f_values (list): List of fitness values corresponding to the individuals in the population.
    - min_tournament_size (int): Minimum number of individuals in each tournament.
    - max_tournament_size (int): Maximum number of individuals in each tournament.

    Returns:
    - selected_parents (list): List of selected parents from the population.
    """
    num_parents = int(len(population)/2)
    selected_parents = []  # List to store the selected parents

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

    #print(len(selected_parents))
    return selected_parents  # Return the list of selected parents


# Returns a survivor array containing surviving children (only!).
# Some (small) number of surviving children are picked based on fitness.
# The rest are picked randomly.
def survivor_selector_mu_lambda(children, no_best_picks):
    survivors = np.random.uniform(-1, 1, (pop_size, n_vars))  # preallocate a random array for survivors
    survivors_f = [0] * pop_size  # preallocate an array for survivors fitness

    children_without_sigma = pop_weights_only(children)

    children_fitness = evaluate(env, children_without_sigma)

    indices_best_children = np.argpartition(children_fitness, -no_best_picks)[-no_best_picks:]

    randints = list(range(0, len(children)))  # Create list of integers(unique)
    random.shuffle(randints)  # Shuffle the list
    for i in range(no_best_picks):  # add some number of best children to the new population
        survivors[i] = children[indices_best_children[i]]
        survivors_f[i] = children_fitness[indices_best_children[i]]  # Save the fitness of the selected child

    for i in range(no_best_picks, pop_size):  # fill the rest of the population with random children
        survivors[i] = children[randints[i]]
        survivors_f[i] = children_fitness[randints[i]]
    return survivors, survivors_f


def find_diverse_indexes(pop1, pop2, n):
    """
    Find the most diverse individuals for migration through:
    1. Calculating the average of the other population
    2. Compare each individual of the other population (diversity score) to this average based on the Euclidean distance
    3. Store the n largest differences for migration

    Args:
    - pop1 (list of arrays): A list containing arrays representing individuals in the population.
    - pop2 (list of arrays): A list containing arrays representing individuals in the population.
    - n (even integer): Desired number of individuals with highest diversity score

    Returns:
    Two lists of indexes for most diverse individuals for pop1 and pop2
    """

    n_p = int(n / 2)
    most_diverse_indices = []

    "1. Calculating the average of both populations"
    avg_pop1 = np.mean(pop1, axis=0)
    avg_pop2 = np.mean(pop2, axis=0)

    "2. Loop over the individuals to find the highest diversity"

    # For population 1
    euclidean_distances1 = np.linalg.norm(pop1 - avg_pop2, axis=1)
    most_diverse_indices1 = np.argsort(euclidean_distances1)[-n_p:]
    most_diverse_indices.append(most_diverse_indices1)

    # For population 2
    euclidean_distances2 = np.linalg.norm(pop2 - avg_pop1, axis=1)
    most_diverse_indices2 = np.argsort(euclidean_distances2)[-n_p:]
    most_diverse_indices.append(most_diverse_indices2)

    return most_diverse_indices


def migration(pop1, pop2, indexes):
    """
    Migrate the most different individuals between two islands' populations.

    Args:
    - pop1 (array of lists): The population of the first island.
    - pop1 (array of lists): The population of the second island.
    - indexes (list of arrays): The indexes of the most diverse individuals in pop1 and pop2 respsectively

    Returns:
    - pop1 (array of lists): Updated population of the first island after migration.
    - pop2 (array of lists): Updated population of the second island after migration.
    """

    # Remove the selected individuals from their original islands
    pop_island_1 = [ind for i, ind in enumerate(pop1) if i not in indexes[0]]
    pop_island_2 = [ind for i, ind in enumerate(pop2) if i not in indexes[1]]

    # Append the migrating individuals to the destination island
    migrating_individuals1 = pop1[indexes[0]]
    migrating_individuals2 = pop2[indexes[1]]

    pop_island_1.extend(migrating_individuals1)
    pop_island_2.extend(migrating_individuals2)

    return np.array(pop_island_1), np.array(pop_island_2)

if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    env.update_parameter('visuals', True)
    evaluate(env, [bsol])

    sys.exit(0)

elif run_mode == 'train':
    for run_number in range(10): #define how many times to run the experiment
        print(f"\nExperiment number: {run_number + 1}")

        #Reinitialize parameters for each of the test runs
        max_f = -1
        avg_f = -1
        low_f = 999

        overall_best = -1
        fitness_avg_history = []
        fitness_best_history = []
        fitness_history = []

        #Generate 266 genes, at loc 0 we find the sigma, the rest of the array is the weights
        pop_isl1 = np.random.uniform(-1, 1, (pop_size, n_vars)) #Initialize population, with extra value for the weights
        pop_isl2 = np.random.uniform(-1, 1, (pop_size, n_vars)) #Initialize population, with extra value for the weights

        #Generate the initial sigma values and place them at location 0 for each individual array
        sigma_vals_i = [random.uniform(sigma_i_U,sigma_i_L) for individual in range(pop_size)]
        pop_isl1[:, 0] = sigma_vals_i
        pop_isl2[:, 0] = sigma_vals_i

        #Evaluate populations : initialization
        pop_f_isl1 = evaluate(env,pop_weights_only(pop_isl1))
        max_f_isl1 = max(pop_f_isl1)
        avg_f_isl1 = sum(pop_f_isl1) / len(pop_f_isl1)

        pop_f_isl2 = evaluate(env,pop_weights_only(pop_isl2))
        max_f_isl2 = max(pop_f_isl2)
        avg_f_isl2 = sum(pop_f_isl2) / len(pop_f_isl2)

        step_size = math.e ** (tao * np.random.normal(0, 1))
        Gen = 0
        
        while Gen < maxGens:
            print(f'Generation: {Gen}')
            
            # Determine the crossover type based on the generation threshold
            if Gen < crossover_threshold:
                crossover_type = 'uniform'
            else:
                crossover_type = 'multipoint'
            
            # Parent selection island 1
            parents_isl1 = []
            parents_isl1 = adaptive_tournament_selection(pop_isl1, pop_f_isl1, 4)  # generates 100 parents - parent selection seems to make the convergence faster
            new_kids_isl1 = np.random.uniform(-1, 1, (N_newGen, n_vars))  # preallocate 600 kids

            # Recombination island 1
            for i in range(0, N_newGen, 2):
                if crossover_type == 'multipoint':
                    baby1, baby2 = multipoint_crossover(parents_isl1[i], parents_isl1[i + 1], mutation_threshold, tao, e0)
                else:
                    baby1, baby2 = uniform_recombination(parents_isl1[i], parents_isl1[i + 1])
                new_kids_isl1[i] = baby1
                new_kids_isl1[i + 1] = baby2

            # Parent selection island 2
            parents_isl2 = []
            parents_isl2 = adaptive_tournament_selection(pop_isl2, pop_f_isl2, 4)  # generates 100 parents - parent selection seems to make the convergence faster
            new_kids_isl2 = np.random.uniform(-1, 1, (N_newGen, n_vars))  # preallocate 600 kids

            # Recombination island 2
            for i in range(0, N_newGen, 2):
                if crossover_type == 'multipoint':
                    baby1, baby2 = multipoint_crossover(parents_isl2[i], parents_isl2[i + 1], mutation_threshold, tao, e0)
                else:
                    baby1, baby2 = uniform_recombination(parents_isl2[i], parents_isl2[i + 1])
                new_kids_isl2[i] = baby1
                new_kids_isl2[i + 1] = baby2

            # Survivor selection island 1
            survivors_isl1, survivors_isl1_f = survivor_selector_mu_lambda(new_kids_isl1, fitness_survivor_no)
            for i in range(pop_size):
                pop_isl1[i] = survivors_isl1[i]

            # Survivor selection island 2
            survivors_isl2, survivors_isl2_f = survivor_selector_mu_lambda(new_kids_isl2, fitness_survivor_no)
            for i in range(pop_size):
                pop_isl2[i] = survivors_isl2[i]

            # Migration every X gens
            if (Gen % migration_multiple == 0) and (Gen != 0):
                diverse_indexes = find_diverse_indexes(pop_isl1, pop_isl2, migrating_indivs_no)
                pop_isl1n, pop_isl2n = migration(pop_isl1, pop_isl2, diverse_indexes)
                pop_isl1, pop_isl2 = pop_isl1n, pop_isl2n

            Gen += 1
            #print(f"popisl1 = {pop_isl1}, with type {type(pop_isl1)}, and length {len(pop_isl1)}")

            pop_without_sigma_isl1 = pop_weights_only(pop_isl1)
            pop_without_sigma_isl2 = pop_weights_only(pop_isl2)

            pop_f_isl1 = survivors_isl1_f
            pop_f_isl2 = survivors_isl2_f
            max_f_isl1 = max(pop_f_isl1)
            max_f_isl2 = max(pop_f_isl2)
            avg_f_isl1 = sum(pop_f_isl1) / len(pop_f_isl1)
            avg_f_isl2 = sum(pop_f_isl2) / len(pop_f_isl2)
            print(f'island 1: {max_f_isl1:.2f}, {avg_f_isl1:.2f}, island 2: {max_f_isl2:.2f}, {avg_f_isl2:.2f}, {crossover_type}')


            if max_f_isl1 > overall_best:
                overall_best = max_f_isl1
                best = np.argmax(pop_f_isl1)
                best_individual = pop_without_sigma_isl1[best]
                best_file_name = f'/best_{run_number}.txt'
                np.savetxt(experiment_name + best_file_name, pop_without_sigma_isl1[best])
            
            if max_f_isl2 > overall_best:
                overall_best = max_f_isl2
                best = np.argmax(pop_f_isl2)
                best_individual = pop_without_sigma_isl2[best]
                best_file_name = f'/best_{run_number}.txt'
                np.savetxt(experiment_name + best_file_name, pop_without_sigma_isl2[best])
            # Store fitness history for each generation
            fitness_avg_history.append((avg_f_isl1+avg_f_isl2)/2)
            fitness_best_history.append(overall_best)

        energyGain=individual_gain(test_env, best_individual)
        
        save_run(fitness_avg_history, fitness_best_history, energyGain, 'best', run_number)
