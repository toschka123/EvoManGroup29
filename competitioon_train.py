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
import optuna
import itertools


# main() includes our adaptive_islanding.py code 11/10/2023 (you can copy past new versions into it)
# below that is the optional tuning part( adjust maxgens and move the variables you want to test as arguments in main )
# changed initial parameters
# fitness based on energygain -800 to 800
# disabled migration&recombination (still "recombines", but always 100% from 1 parent)

def main(
    # <editor-fold desc="Variables we might tune (end with a comma)">
    island_no=2,  # must be even
    migrating_indivs_no = 5,  # how many inidivduals migrate
    migration_multiple = 3,  # migration occurs every X gens
    mutation_threshold=0.04,
    tau = 1
    # </editor-fold>
):
    # <editor-fold desc="Initialise remaining variables">
    pop_size = int(100 / island_no)
    max_f = -1
    avg_f = -1
    low_f = 999
    maxGens = 1000
    Gen = 0
    N_newGen = pop_size * 4  # define how many offsprings we want to produce and how many old individuals we want to kill NOTE This has to be even!!
    fitness_survivor_no = int(40 /island_no)  # how many children in the new generation will be from "best". The rest are random.
    gaussian_mutation_sd = 0.5
    overall_best = -1
    e0 = 0.000001               #Formulate the boundary condition for sigma'
    #COMPLETELY RANDOM NR NOW !!
    run_mode = "train"

    fitness_avg_history = []
    fitness_best_history = []
    fitness_history = []

    #Define the bounds of your initial sigma values and tau for self-adaptive mutation
    sigma_i_U = 0.7
    sigma_i_L = 0.7

    #Generate the stepsize (mutation size) of your sigma value
    step_size = 0.95
    # </editor-fold>
    crossover_threshold = 10
    # <editor-fold desc="Simulation settings">
    # runs simulation
    def simulation(env,x):
        f,p,e,t = env.play(pcont=x)
        return f

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
                      enemies=[1],
                      multiplemode="no",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),  # you  can insert your own controller here
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    # number of variables for multilayer with 10 hidden neurons (265) plus one sigma value
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5 + 1
    # </editor-fold>

    # <editor-fold desc="Functions">
    # evaluation
    def evaluate(env, x):
        return np.array(list(map(lambda y: simulation(env,y), x)))

    def evaluate_8(pop):
        scores = []
        beats = []
        for individual in pop:
            energy_gains = []
            beat = 0
            for i in range(8):
                env = Environment(experiment_name=experiment_name,
                                  enemies=[i+1],
                                  multiplemode="no",
                                  playermode="ai",
                                  player_controller=player_controller(n_hidden_neurons),
                                  # you  can insert your own controller here
                                  enemymode="static",
                                  level=2,
                                  speed="fastest",
                                  visuals=False)
                f, p, e, t = env.play(pcont=individual)
                energy_gain = int(p) - int(e)
                if energy_gain > 0:
                    beat += 1
                energy_gains.append(energy_gain)
            beats.append(beat)
            score = 0
            for gain in energy_gains:
                score += gain
            scores.append(score)
        score1 = scores[:math.floor(len(scores)/2)]
        score2 = scores[math.floor(len(scores)/2):]
        print(f"scores: {score1}")
        print(f"scores: {score2}")
        print(f"beats: {beats}")
        return scores


    def evaluate_gain(env, individual):
        return np.array((map(lambda y: individual_gain(env, y),individual)))


    def individual_gain(env, individual):
        f,p,e,t = env.play(pcont=individual)
        indiv_gain = int(p)-int(e)
        return indiv_gain


    # Function that returns the population with only the 265 weights, no sigma
    def pop_weights_only(pop):
        weights_only = pop[:,1:]
        return weights_only


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
            baby1.append(i1[i])
            baby2.append(i2[i])
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

        return selected_parents  # Return the list of selected parents

    def multipoint_crossover(parent1, parent2, mutation_threshold, tau, e0):
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
            step_size = math.e ** (tau * random.normalvariate(0, 1))
            
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
            step_size = math.e ** (tau * random.normalvariate(0, 1))
            
            # Calculate the mutated sigma value
            sigma_prime2 = baby2[0] * step_size
            
            # Ensure the mutated sigma value does not go below e0
            if sigma_prime2 < e0:
                sigma_prime2 = e0
            
            # Update baby2's sigma value
            baby2[0] = sigma_prime2

        # Return the resulting offspring, baby1 and baby2
        return baby1, baby2

    # Returns a survivor array containing surviving children (only!).
    # Some (small) number of surviving children are picked based on fitness.
    # The rest are picked randomly.
    def survivor_selector_mu_lambda(children, no_best_picks):
        survivors = np.random.uniform(-1, 1, (pop_size, n_vars)) #preallocate a random array for survivors
        survivors_f = [0] * pop_size #preallocate a random array for survivors

        children_without_sigma = pop_weights_only(children)

        #print(f"children: {children}")
        children_fitness = evaluate_8(children_without_sigma)
        #print(f"children fitness: {children_fitness}")

        indices_best_children = np.argpartition(children_fitness, -no_best_picks)[-no_best_picks:]

        randints = list(range(0, len(children)))
        random.shuffle(randints)
        #print(f"best index: {indices_best_children}")
        #print(f"ints: {randints}")
        for i in range(no_best_picks): #add some number of best children to the new population
            randints.remove(indices_best_children[i])
            #print(f"ints: {randints}")
            survivors[i] = children[indices_best_children[i]]
            #print(f"surv i: {survivors_f[i]}")
            #print(f"child i: {children_fitness[indices_best_children[i]]}")
            survivors_f[i] = children_fitness[indices_best_children[i]]
            #print(f"survivors elite 20: {survivors}")
            #print(f"survivors elite f: {survivors_f}")

        for i in range(no_best_picks, pop_size): #fill the rest of the population with random children
            survivors[i] = children[randints[i]]
            survivors_f[i] = children_fitness[randints[i]]
            #print(f"survivors 30: {survivors}")
            #print(f"survivors f: {survivors_f}")
        return survivors, survivors_f


    # Exchanges some number of individuals between two islands
    def migrate(pop_island_1, pop_island_2, no_individuals):
        indiv_indices = random.sample(range(1, pop_size), no_individuals) #exchange randomly. TODO: exchange most different indivs
        leaving_isl_1 = pop_island_1[indiv_indices]
        leaving_ils_2 = pop_island_2[indiv_indices]

        pop_island_1[indiv_indices] = leaving_ils_2
        pop_island_2[indiv_indices] = leaving_isl_1

        print('migrated')

        return pop_island_1, pop_island_2
    # </editor-fold>

    # <editor-fold desc="Running the code">
    if run_mode =='test':

        bsol = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        env.update_parameter('visuals', True)
        evaluate_8([bsol])

        sys.exit(0)

    elif run_mode == 'train':
      for run_number in range(1): #define how many times to run the experiment
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
        pop_f_isl1 = evaluate_8(pop_weights_only(pop_isl1))
        max_f_isl1 = max(pop_f_isl1)
        avg_f_isl1 = sum(pop_f_isl1) / len(pop_f_isl1)

        pop_f_isl2 = evaluate_8(pop_weights_only(pop_isl2))
        max_f_isl2 = max(pop_f_isl2)
        avg_f_isl2 = sum(pop_f_isl2) / len(pop_f_isl2)
        print(f'INIT: island 1: {max_f_isl1:.2f}, {avg_f_isl1:.2f}, island 2: {max_f_isl2:.2f}, {avg_f_isl2:.2f}\n')

        Gen = 1

        while True:
            print(f'Generation: {Gen}')
            sigmas = []
            for bb in pop_isl1:
                sigmas.append(bb[0])
            for bb in pop_isl2:
                sigmas.append(bb[0])
            minsigmas = min(sigmas)
            maxsigmas = max(sigmas)
            avgsigma = sum(sigmas) / len(sigmas)
            print(f"avg sigma: {avgsigma} max: {maxsigmas} min: {minsigmas}")
            #Parent selection island 1
            parents_isl1 = []
            parents_isl1 = adaptive_tournament_selection(pop_isl1, pop_f_isl1, 4) #generates 100 parents - parent selection seems to make the convergion faster
            new_kids_isl1 = np.random.uniform(-1, 1, (N_newGen, n_vars)) #preallocate 600 kids

            # Determine the crossover type based on the generation threshold
            if Gen < crossover_threshold:
                crossover_type = 'uniform'
            else:
                crossover_type = 'multipoint'

            # Recombination island 1
            for i in range(0, N_newGen, 2):
                if crossover_type == 'multipoint':
                    baby1, baby2 = multipoint_crossover(parents_isl1[i], parents_isl1[i + 1], mutation_threshold, tau, e0)
                else:
                    baby1, baby2 = uniform_recombination(parents_isl1[i], parents_isl1[i + 1])
                new_kids_isl1[i] = baby1
                new_kids_isl1[i + 1] = baby2

            #Parent selection island 2
            parents_isl2=[]
            parents_isl2 = adaptive_tournament_selection(pop_isl2, pop_f_isl2, 4) #generates 100 parents - parent selection seems to make the convergion faster
            new_kids_isl2 = np.random.uniform(-1, 1, (N_newGen, n_vars)) #preallocate 600 kids

            # Recombination island 2
            for i in range(0, N_newGen, 2):
                if crossover_type == 'multipoint':
                    baby1, baby2 = multipoint_crossover(parents_isl2[i], parents_isl2[i + 1], mutation_threshold, tau, e0)
                else:
                    baby1, baby2 = uniform_recombination(parents_isl2[i], parents_isl2[i + 1])
                new_kids_isl2[i] = baby1
                new_kids_isl2[i + 1] = baby2

            #Survivor selection island 1
            survivors_isl1, pop_f1 = survivor_selector_mu_lambda(new_kids_isl1, fitness_survivor_no)
            for i in range(pop_size):
                pop_isl1[i] = survivors_isl1[i]

            #Survivor selection island 2
            survivors_isl2, pop_f2 = survivor_selector_mu_lambda(new_kids_isl2, fitness_survivor_no)
            for i in range(pop_size):
                pop_isl2[i] = survivors_isl2[i]

            # Migration every X gens
            if (Gen % migration_multiple == 0) and (Gen != 0):
                diverse_indexes = find_diverse_indexes(pop_isl1, pop_isl2, migrating_indivs_no)
                pop_isl1n, pop_isl2n = migration(pop_isl1, pop_isl2, diverse_indexes)
                pop_isl1, pop_isl2 = pop_isl1n, pop_isl2n

            Gen+=1

            pop_without_sigma_isl1 = pop_weights_only(pop_isl1)
            pop_without_sigma_isl2 = pop_weights_only(pop_isl2)

            pop_f_isl1 = pop_f1
            pop_f_isl2 = pop_f2
            max_f_isl1 = max(pop_f_isl1)
            max_f_isl2 = max(pop_f_isl2)
            avg_f_isl1 = sum(pop_f_isl1) / len(pop_f_isl1)
            avg_f_isl2 = sum(pop_f_isl2) / len(pop_f_isl2)
            print(f'island 1: {max_f_isl1:.2f}, {avg_f_isl1:.2f}, island 2: {max_f_isl2:.2f}, {avg_f_isl2:.2f}, {crossover_type}')


            if max_f_isl1 > overall_best:
                overall_best = max_f_isl1
                best = np.argmax(pop_f_isl1)
                best_individual = pop_without_sigma_isl1[best]
                overall_best = max_f_isl1
                np.savetxt(experiment_name + '/best.txt', pop_without_sigma_isl1[best])
            if max_f_isl2 > overall_best:
                overall_best = max_f_isl2
                best = np.argmax(pop_f_isl2)
                best_individual = pop_without_sigma_isl2[best]
                overall_best = max_f_isl2

                np.savetxt(experiment_name + '/best.txt', pop_without_sigma_isl2[best])
            # Store fitness history for each generation
            fitness_avg_history.append((avg_f_isl1+avg_f_isl2)/2)
            fitness_best_history.append(overall_best)
            # Calculate and store the fitness values of the current population
            #fitness_values = evaluate(env, pop_without_sigma)
            #fitness_history.append(fitness_values)

            # Calculate the standard deviation of fitness values
            #fitness_std = np.std(fitness_values)

            # Print or log the fitness diversity metric for the current generation
            #print(f"Generation {Gen}: Fitness Diversity (Std Dev): {fitness_std}")

        #avg_sigma_end = sum(pop[:,1])/len(pop[:,1])


        energyGain=individual_gain  (env, best_individual)
        print(f"energygain: {energyGain}")
        #print(energyGain)
        save_run(fitness_avg_history, fitness_best_history, 0, 0,energyGain, "heatmen", run_number)
        #save_run(fitness_avg_history, fitness_best_history, avg_sigma_start, avg_sigma_end)
        # After the loop, you can visualize the fitness diversity over generations if needed
        """plt.plot(range(maxGens), [np.std(fitness) for fitness in fitness_history])
        plt.title("Fitness Diversity Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Standard Deviation of Fitness")
        plt.show()"""
    return("insert best score here")
    # </editor-fold>


# <editor-fold desc="Parameter tuning">
tuning = False

if not tuning:
    main()

else:
    search_space = {
        # variables to tune : values to try
        'tau': [0.05, 1, 1.41],
        'mutation_threshold': [0.04, 0.06, 0.08]
    }

    values_to_try = []
    for key in search_space.items():
        values_to_try.append(key[1])
    combinations = list(itertools.product(*values_to_try))
    for combination in combinations:
        search = {key[0]: combination[i] for i, key in enumerate(search_space.items())}

        print("starting a test run with:")
        for key, value in search.items():
            print(key, value)
        print(f"best score of this run: {main(**search)}")
# </editor-fold>

