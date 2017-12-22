import numpy as np
import random
import pdb
import sys
from k_means import KMeans
from hac import HAC

# Set seed
np.random.seed([27192759])

def init_population(size_of_population, num_of_features):
    '''Use a uniform distribution to create a random collection
    of candidate_feature_sets'''
    feature_sets_to_accuracy = []
    population_size = 0
    # while we still need to populate...
    while len(feature_sets_to_accuracy) < size_of_population:
        # create a random bit string of length = num of features
        candidate_feature_set = \
            [1 if random.random() > 0.5 else 0 for feature in range(num_of_features)]
        # only add bit string if it hasn't been added yet and isn't all zeros
        if (sum(candidate_feature_set) > 0) and \
            (not candidate_feature_set in feature_sets_to_accuracy):
            feature_sets_to_accuracy.append((candidate_feature_set, 0.0))
    return feature_sets_to_accuracy

def evaluate_model(model, model_type, num_of_classes, candidate_feature_set, data_set):
    '''This method uses the inputted feature subset to cluster the inputted data and
    scores performance using a LDA-like objective function.'''
    # Convert candidate_feature_set representation from
    # f_1, ... f_d to the list of indices of the f_i = 1
    # (for example, [1 0 0 1 0] -> [0 3]
    candidate_feature_set = \
        [idx for idx in range(len(candidate_feature_set)) if candidate_feature_set[idx] == 1]
    if model_type == "Kmeans":
        model = KMeans(num_of_classes)
    elif model_type == "HAC":
        model = HAC(num_of_classes)
    model.cluster(data_set[:,candidate_feature_set])
    return model.calculate_performance() 

def select_parents(feature_sets_to_accuracy):
    '''This method selects from the population two parents based on 
    their fitness proportions.'''
    candidate_feature_sets = []
    total_fitness = 0.0
    # sum total fitness of all feature sets (used to determine fitness proportion for each subset)
    for feature_set in feature_sets_to_accuracy:
        candidate_feature_sets.append(feature_set[0])
        total_fitness += feature_set[1]
    fitness_proportions = [(feature_set[1] / total_fitness) for feature_set in feature_sets_to_accuracy]
    # choose parents with probability proportional to fitness proportions
    parents = np.random.choice(range(len(candidate_feature_sets)), size=2, replace=True, p=fitness_proportions)
    return candidate_feature_sets[parents[0]], candidate_feature_sets[parents[1]]

def crossover(parent1, parent2):
    '''This method chooses a crossover point randomly and crosses over the inputted parents.'''
    crossover_idx = random.randrange(0, len(parent1))
    new_parent_1 = parent1[0:crossover_idx]  + parent2[crossover_idx:]
    new_parent_2 = parent2[0:crossover_idx]  + parent1[crossover_idx:]
    return new_parent_1, new_parent_2

def mutate(offspring):
    '''This method mutates the offspring with a small probability.'''
    mutated_offspring = [abs(offspring[idx] - 1) if random.random() <= 0.01 else offspring[idx] for idx in range(len(offspring))]
    if sum(mutated_offspring) == 0:
        mutated_offspring[random.randrange(0, len(offspring))] = 1
    return mutated_offspring

def perform_GA_feature_selection(model, model_type, num_of_classes, data_set):
    # initialize population (randomized feature sets), 
    # Structure: list of (candidate, model accuracy))
    feature_sets_to_accuracy = init_population(10, data_set.shape[1] - 1)

    # evaluate initial population of candidiate feature sets accuracy 
    for candidate_idx in range(len(feature_sets_to_accuracy)):
        feature_sets_to_accuracy[candidate_idx] = \
            (feature_sets_to_accuracy[candidate_idx][0],
            evaluate_model(
                model, 
                model_type, 
                num_of_classes,
                feature_sets_to_accuracy[candidate_idx][0], 
                data_set))

    # run 100 trials
    for i in range(100):
        print("GA Trial: %d" % i)
        '''1. select subset of population
            a. fitness proportionate selection
               i. using two parents
              ii. accuracy is the probability a feature set is chosen WITH REPLACEMENT
                  a. P(X_i) = f(X_i) / sum_{j=1}^{n} f(X_j)'''
        parent1, parent2 = select_parents(feature_sets_to_accuracy)

        '''2. recombine for new offspring
           a. Crossover
               i. randomly select one feature as crossover point'''
        offspring1, offspring2 = crossover(parent1, parent2)

        '''3. mutate offspring
           a. Flip bits with probability of 0.01'''
        offspring1 = mutate(offspring1)
        offspring2 = mutate(offspring2)

        '''4. evaluate offspring fitness by running model'''
        offspring1 = \
            (offspring1, evaluate_model(model, model_type, num_of_classes, offspring1, data_set))
        offspring2 = \
            (offspring2, evaluate_model(model, model_type, num_of_classes, offspring2, data_set))

        '''5. use evalution to determine how to update population
            a. steady state replacement, replace number of offspring generated
            b. update fitness proportion'''
        feature_sets_to_accuracy = sorted(feature_sets_to_accuracy, key=lambda tup: tup[1])
        bottom_performing_offspring = \
            [offspring1, offspring2, feature_sets_to_accuracy[-1], feature_sets_to_accuracy[-2]]
        bottom_performing_offspring = sorted(bottom_performing_offspring, key=lambda tup: tup[1])
        feature_sets_to_accuracy = feature_sets_to_accuracy[:-2] + [bottom_performing_offspring[-1], bottom_performing_offspring[-2]]
    return sorted(feature_sets_to_accuracy, key=lambda tup: tup[1])[0]
