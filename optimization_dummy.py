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
run_mode = "train"
pop_size = 100
max_f =-1
avg_f =-1
low_f = 999
maxGens=20
Gen=0
N_newGen=pop_size # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
mutation_strength = 0.04
overall_best = -1

pop = np.random.uniform(-1, 1, (pop_size, n_vars)) #initialize population
pop_f = evaluate(env,pop) #evaluate population
max_f=max(pop_f)
avg_f = sum(pop_f)/len(pop_f)
low_f = min(pop_f)
print(max_f, avg_f)

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
    mutation = np.random.normal(0, 0.5)

    if mutation + gene > 1: #if values too big
        return 0.99
    elif mutation < -1: #if values too small
        return -0.99

    gene += mutation

    return gene


"""def parent_selection(population, f_vals): #random 50, change to something sensible later
    list_of_parents=[]
    for i in range(20):
        ip1 = randint(0,len(population)-1)
        ip2 = randint(0,len(population)-1)
        p1 = population[ip1]
        p2 = population[ip2]
        list_of_parents.append([p1, p2])
    return list_of_parents"""

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


if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)
else:
    while Gen < maxGens:
        parents=[]
        for i in range(int(N_newGen/2)): #Choose how many pairs of parents we want
            parents.append(parent_selection(pop, pop_f,4, N_newGen))
        new_kids = []
        for pairs in parents: #Each pair of parents generates two offspring
            baby1, baby2 = npoint_recombination(pairs[0], pairs[1],2)
            new_kids.append(baby1)
            new_kids.append(baby2)

        #This is combining new and old and
        """old_generation = pop.tolist()
        total = old_generation + new_kids
        total =np.array(total)
        total_f = evaluate(env, total)
        survivors = select_surv(total, total_f, 100)
        for i in range(len(survivors)):
            pop[i] = survivors[i]"""

        #This is without survivor select - age based
        inds = kill_people(pop, N_newGen) # pick how many individuals we want to kill
        for i in range(len(inds)):
            pop[inds[i]] = new_kids[i]


        Gen+=1
        pop_f = evaluate(env,pop) #evaluate new population
        max_f = max(pop_f)
        avg_f = sum(pop_f) / len(pop_f)
        low_f = min(pop_f)
        print(max_f, avg_f,len(pop))

        if max_f > overall_best:
            best = np.argmax(pop_f)
            best_individual = pop[best]
            overall_best = max_f
            np.savetxt(experiment_name + '/best.txt', pop[best])



