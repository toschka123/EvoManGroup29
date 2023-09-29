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
import csv
import datetime

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def save_run_no_selfadapt(mean_f, best_f):
    # giving the file a timestamp
    time = datetime.datetime.now().strftime("%d%H%M")
    filename = f'evcomprun_no_self_adapat_{time}.csv'

    with open(filename, mode='w', newline='') as file:
        f = csv.writer(file)
        f.writerow(['Mean fitness', 'Best fitness'])
        for i in range(len(mean_f)):
            f.writerow([mean_f[i], best_f[i]])


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
                enemies=[7],
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
max_f = -1
avg_f = -1
low_f = 999
maxGens = 2
Gen = 0
N_newGen = pop_size * 4  # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
mutation_strength = 0.04
fitness_survivor_no = 20  # how many children in the new generation will be from "best". The rest are random.
gaussian_mutation_sd = 0.5
fitness_avg_history = []
fitness_best_history = []
overall_best = -1
fitness_history = []

run_mode = "train"

def random_points(n):
    crossover_list = []
    for i in range(n):
        # generate random point in the list of weights
        k = random.randint(0, 265)
        # checking the point is not already in the list
        if k not in crossover_list:
            crossover_list.append(k)

    crossover_list.sort()
    return crossover_list

def crossover (p1, p2, point):
        for i in range(point, len(p1)):
            p1[i], p2[i] = p2[i], p1[i]
        return p1, p2

#Uniform recombination
def uniform_recombination(i1, i2): #Takes as input two parents and returns 2 babies, in each position 50% chance to have parent1's gene
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

#n-point crossover
def npoint_recombination(i1, i2, n):
    # find the crossover point locations
    crossover_locs = random_points(n)

    # track the crossover points
    swapped = False

    #loop over weights and swap weights between parents until you encounter the next crossover location
    for i in range(len(i1)):
        if swapped:
            if i in crossover_locs:
                swapped = False
            else:
                i1[i], i2[i] = i2[i], i1[i]
        elif not swapped and i in crossover_locs:
            swapped = True

    return i1, i2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_strength:
            individual[i] += random.uniform(-1, 1)  # You can adjust the mutation range
    return individual


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

        # Calculate the diversity score for each selected individual
        """for index in tournament_indices:
            # Calculate the absolute differences between the fitness of the selected individual
            # and the fitness of other individuals in the tournament, then take the mean.
            diversity_scores[index] += np.mean(np.abs(tournament_fitness - f_values[index]))"""

        # Choose the best individual from the tournament as the parent
        tournament_indices = np.random.choice(num_parents, size=max_tournament_size, replace=False)
        # Calculate the fitness values of the selected individuals
        tournament_fitness = [f_values[i] for i in tournament_indices]
        best_index2 = tournament_indices[np.argmax(tournament_fitness)]

        # Append the best individual to the list of selected parents
        selected_parents.append(population[best_index1])
        selected_parents.append(population[best_index2])

        #second_parent_ind = random.randint(0, 99)
        #selected_parents.append(population[second_parent_ind])

        """# Update tournament size for the next selection (adaptive)
        if current_tournament_size < max_tournament_size:
            current_tournament_size += tournament_size_increment"""

    #print(len(selected_parents))
    return selected_parents  # Return the list of selected parents


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

if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    env.update_parameter('visuals', True)
    evaluate(env, [bsol])

    sys.exit(0)

elif run_mode == 'train':


    pop = np.random.uniform(-1, 1, (pop_size, n_vars))  # initialize population
    pop_f = evaluate(env, pop)  # evaluate population
    max_f = max(pop_f)
    avg_f = sum(pop_f) / len(pop_f)
    low_f = min(pop_f)
    print(max_f, avg_f)

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


        survivors = survivor_selector_mu_lambda(new_kids, fitness_survivor_no)
        for i in range(pop_size):
            pop[i] = survivors[i]

        Gen+=1
        pop_f = evaluate(env,pop) #evaluate new population
        max_f = max(pop_f)
        avg_f = sum(pop_f) / len(pop_f)
        low_f = min(pop_f)
        print(max_f, avg_f)


        fitness_avg_history.append((avg_f))

        if max_f > overall_best:
            overall_best = max_f
            best = np.argmax(pop_f)
            best_individual = pop[best]
        fitness_best_history.append(overall_best)

# saves simulation state
np.savetxt(experiment_name + '/best.txt', best_individual)
save_run_no_selfadapt(fitness_avg_history, fitness_best_history)

solutions = [pop, pop_f]
env.update_solutions(solutions)
env.save_state()

        # Store fitness history for each generation

       # Calculate and store the fitness values of the current population
"""        fitness_values = evaluate(env, pop)
        fitness_history.append(fitness_values)

        # Calculate the standard deviation of fitness values
        fitness_std = np.std(fitness_values)"""

        # Print or log the fitness diversity metric for the current generation
"""        print(f"Generation {Gen}: Fitness Diversity (Std Dev): {fitness_std}")

    # After the loop, you can visualize the fitness diversity over generations if needed
    plt.plot(range(maxGens), [np.std(fitness) for fitness in fitness_history])
    plt.title("Fitness Diversity Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Standard Deviation of Fitness")
    plt.show()"""


