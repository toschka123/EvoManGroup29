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


# number of variables for multilayer with 10 hidden neurons (265) plus one sigma value
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 +1

# start writing your own code from here
pop_size = 100
max_f = -1
avg_f = -1
low_f = 999
maxGens = 20
Gen = 0
N_newGen = pop_size * 4  # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
mutation_threshold = 0.04
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
sigma_i_U = 0.5  # when starting at 1.0 scores dont really improve until it reaches around 0.5
sigma_i_L = 0.5  # 0.1 could be good too, but maybe converges too quick? 0.01 definitely too refined to start with

#Generate the stepsize (mutation size) of your sigma value
tao = 0.05
step_size = 0.95  # mutation slowly becomes smaller
#step_size = math.e ** (tao * np.random.normal(0, 1))

#Decide which baby to mutate


#Generate the initial sigma values and place them at location 0 for each individual array
sigma_vals_i = [random.uniform(sigma_i_U,sigma_i_L) for individual in range(pop_size)]
pop[:, 0] = sigma_vals_i
avg_sigma_start = sum(sigma_vals_i)/len(sigma_vals_i)

#Evaluate population
pop_f = evaluate(env,pop_weights_only(pop))
max_f = max(pop_f)
avg_f = sum(pop_f)/len(pop_f)
low_f = min(pop_f)
print(max_f, avg_f)
run_mode = "train"

#Uniform recombination
def uniform_recombination(i1, i2): #Takes as input two parents and returns 2 babies, in each position 50% chance to have parent1's gene
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


def mutate_gene_gaussian(gene):
    mutation = np.random.normal(0, gaussian_mutation_sd)

    while (mutation + gene > 1) or (mutation + gene < -1):
        mutation = np.random.normal(0, gaussian_mutation_sd)

    gene += mutation
    return gene

def adaptive_tournament_selection(population, f_values, min_tournament_size=2, max_tournament_size=5):
    num_parents = int(len(population)/2)
    selected_parents = []  # List to store the selected parents

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

    #print(len(selected_parents))
    return selected_parents  # Return the list of selected parents


# Returns a survivor array containing surviving children (only!).
# Some (small) number of surviving children are picked based on fitness.
# The rest are picked randomly.
def survivor_selector_mu_lambda(children, no_best_picks):
    survivors = np.random.uniform(-1, 1, (pop_size, n_vars)) #preallocate a random array for survivors
    survivors_f = [0] * pop_size  # preallocate an array for survivors fitness

    children_without_sigma = pop_weights_only(children)

    children_fitness = evaluate(env, children_without_sigma)

    indices_best_children = np.argpartition(children_fitness, -no_best_picks)[-no_best_picks:]

    randints = list(range(0, len(children)))  # Create list of integers(unique)
    random.shuffle(randints)                  # Shuffle the list
    for i in range(no_best_picks): #add some number of best children to the new population
        survivors[i] = children[indices_best_children[i]]
        survivors_f[i] = children_fitness[indices_best_children[i]]  # Save the fitness of the selected child

    for i in range(no_best_picks, pop_size): #fill the rest of the population with random children
        survivors[i] = children[randints[i]]
        survivors_f[i] = children_fitness[randints[i]]
    return survivors, survivors_f




if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    env.update_parameter('visuals', True)
    evaluate(env, [bsol])

    sys.exit(0)

elif run_mode == 'train':
  for run_number in range(10): #define how many times to run the experiment
    #Reinitialize parameters for each of the test runs
    max_f = -1
    avg_f = -1
    low_f = 999

    overall_best = -1
    fitness_avg_history = []
    fitness_best_history = []
    fitness_history = []

    #Generate 266 genes, at loc 0 we find the sigma, the rest of the array is the weights
    pop = np.random.uniform(-1, 1, (pop_size, n_vars)) #Initialize population, with extra value for the weights

    Gen = 1
    
    while Gen < maxGens:
        # parents = []
        # for i in range(int(N_newGen/2)):
        #     parents.append(adaptive_tournament_selection(pop, pop_f, 6, N_newGen))

        parents=[]
        parents = adaptive_tournament_selection(pop, pop_f, 4) #generates 100 parents - parent selection seems to make the convergion faster
        new_kids = np.random.uniform(-1, 1, (N_newGen, n_vars)) #preallocate 600 kids

        for i in range(0,N_newGen,2):
            baby1, baby2 = uniform_recombination(parents[i], parents[i+1])
            new_kids[i] = baby1
            new_kids[i + 1] = baby2


        """if len(new_kids) > 100:
            for i in range(0,len(new_kids)-100, 2):
                baby1, baby2 = uniform_recombination(parents[randint(0,99)], parents[randint(0,99)])
                new_kids[i+100] = baby1
                new_kids[i+101] = baby2"""

        #print(f"new_kids: {new_kids}")
        #print(f"shape: {len(new_kids), len(new_kids[0])}")
        #print(f"fitness_survivor_no: {fitness_survivor_no}")
        survivors, survivors_f = survivor_selector_mu_lambda(new_kids, fitness_survivor_no)
        for i in range(pop_size):
            pop[i] = survivors[i]

        Gen+=1

        pop_without_sigma = pop_weights_only(pop)

        pop_f = survivors_f #evaluate new population
        max_f = max(pop_f)
        avg_f = sum(pop_f) / len(pop_f)
        low_f = min(pop_f)
        print(max_f, avg_f)


        fitness_avg_history.append((avg_f))
        fitness_best_history.append(max_f)

        if max_f > overall_best:
            overall_best = max_f
            best = np.argmax(pop_f)
            best_individual = pop_without_sigma[best]
            overall_best = max_f

            np.savetxt(experiment_name + '/best.txt', pop_without_sigma[best])
        # Store fitness history for each generation

        # Calculate and store the fitness values of the current population
        fitness_values = evaluate(env, pop_without_sigma)
        fitness_history.append(fitness_values)

        # Calculate the standard deviation of fitness values
        fitness_std = np.std(fitness_values)

        # Print or log the fitness diversity metric for the current generation
        print(f"Generation {Gen}: Fitness Diversity (Std Dev): {fitness_std}")

    avg_sigma_end = sum(pop[:,1])/len(pop[:,1])
    energyGain=individual_gain  (env, pop_without_sigma[best])
    #print(energyGain)
    save_run(fitness_avg_history, fitness_best_history ,energyGain, "heatmen", run_number)
    #save_run(fitness_avg_history, fitness_best_history, avg_sigma_start, avg_sigma_end)
    # After the loop, you can visualize the fitness diversity over generations if needed
    """plt.plot(range(maxGens), [np.std(fitness) for fitness in fitness_history])
    plt.title("Fitness Diversity Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Standard Deviation of Fitness")
    plt.show()"""


