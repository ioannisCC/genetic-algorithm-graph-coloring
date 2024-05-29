import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations
import os

def get_valid_number(prompt):

    while True:
        try:
            value = int(input(prompt))
            if value < 1:
                print("Please enter a positive integer. ")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a positive integer. ")


def get_probability(prompt):

    while True:
        try:
            value = float(input(prompt))
            if value > 1 or value < 0:
                print("Please enter a probability between 0 and 1")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a positive integer. ")


def smallest_power_of_2(n):

    if n <= 0:
        return None  # only positive numbers are considered
    power = 1
    exponent = 0
    while power < n:
        power *= 2
        exponent += 1
    return power, exponent


# generate population
def generate_population(size: int, power: int, dimensions: int): # size for the # of the generated chromosomes
    
    population = []                                              # power for the of bits needed to represent the colors
    for i in range(size):                                        # dimensions for the dimensions of the adjacency matrix (# of graph nodes)
        binary_number = []
        for j in range(power*dimensions):
            binary_number.append(str(random.randint(0,1)))
        population.append(''.join(binary_number))
    return population
# blue: 00, red: 01, green: 10, yellow: 11


def split_chromosomes(power: int, population: list):

    splitted_chromosomes = []
    for i in range(len(population)):
        current_chromosome = population[i]
        splitted_chromosome = [current_chromosome[i:i+power] for i in range(0, len(current_chromosome), power)]
        splitted_chromosomes.append(splitted_chromosome)
    # each pair of two bits represents the color of the node with the corresponding index in the array

    print(splitted_chromosomes)
    return splitted_chromosomes


def assign_colors(splitted_chromosomes: list[int]):

    # initialize color array with 0s
    n = len(splitted_chromosomes)
    m = len(splitted_chromosomes[0])
    colors_matrix = [[0 for _ in range(m)] for _ in range(n)]

    # assign colors accordingly
    for i in range(n):
        for j in range(m):
            if splitted_chromosomes[i][j] == '00':
                colors_matrix[i][j] = 'blue'
            elif splitted_chromosomes[i][j] == '01':
                colors_matrix[i][j] = 'red'
            elif splitted_chromosomes[i][j] == '10':
                colors_matrix[i][j] = 'green'
            else:
                colors_matrix[i][j] = 'yellow'
    # make this generalised by asking user for the represenation of colors for more 2 bits per color           

    print('colorsMatrix '+str(colors_matrix))
    return colors_matrix

#         χ(G) ≤ Δ(G) + 1

# To define a fitness function, we assign a penalty of 1 for every edge that has the same colored vertices incident on it.
#  penalty(i, j) = 1 if there is an edge between i and j
#                    =​ 0 Otherwise
# Thus the fitness of an individual will be the sum of this penalty function over all the edges of the graph.
#          fitness = ∑ penalty(i, j) 

# minimization problem (purpose to find minimum fitness function score)
def fitness(power: int, population: list, adjacency_matrix: list):

    fitness_score_array = []
    splitted_chromosomes = split_chromosomes(power=power, population=population)
    print(splitted_chromosomes)
    colors_matrix = assign_colors(splitted_chromosomes=splitted_chromosomes)

    # if len(colors_matrix) <= len(np_matrix):
    #     for i in range(len(colors_matrix)):
    #         fitness_score = 0
    #         for j in range(len(colors_matrix[i])):
    #             # check if nodes are adjacent
    #             if np_matrix[i][j] == 1:
    #                 # check if colors of adjacent nodes are the same
    #                 if colors_matrix[i][j] != colors_matrix[i][i]:
    #                     fitness_score += 1 # if not the same then increment the score
    #                 else:
    #                     continue
    #         fitness_score_array.append(fitness_score)
    # else:
        # for i in range(len(np_matrix)):
        #     print('i'+str(i))
        #     fitness_score = 0
        #     for j in range(len(np_matrix[0])):
        #         # check if nodes are adjacent
        #         if np_matrix[i][j] == 1:
        #             for k in range(len(colors_matrix[[0]])):
        #                 # check if colors of adjacent nodes are the same
        #                 if colors_matrix[j][k] != colors_matrix[i][k]:
        #                     fitness_score += 1 # if not the same then increment the score
        #                 else:
        #                     continue
        #     fitness_score_array.append(fitness_score)
        # i=0
        # while i < len(colors_matrix):
        #     print('i'+str(i))
        #     fitness_score = 0
        #     for j in range(len(np_matrix)):
        #         for k in range(len(np_matrix[0])):
        #             # check if nodes are adjacent
        #             if np_matrix[j][k] == 1:
        #                 if colors_matrix[i][j] != colors_matrix[i][k]:
        #                     fitness_score += 1 # if not the same then increment the score
        #                 else:
        #                     continue
        #             else:
        #                 continue
        #     fitness_score_array.append(fitness_score)
        #     i+=1

    for i in range(len(colors_matrix)):
        fitness_score = 0

        for j in range(len(adjacency_matrix)):
            for k in range(j+1, (len(adjacency_matrix))):
                # check if nodes are adjacent
                if adjacency_matrix[j,k] == 1 and colors_matrix[i][j] == colors_matrix[i][k]: # if the same then increment the score
                   fitness_score += 1
                else: # keep the score at it was
                    fitness_score = fitness_score
        fitness_score_array.append(fitness_score)

    return fitness_score_array, colors_matrix


# select best solutions (parents) from the current generation to descend to the following ones
def tournament_selection(population: list, fitness_score_array: list, k: int): # k refers to the # of values compared each time

    indices = [i for i in range(len(fitness_score_array))] # will save th indices of the fitness_score_array
    best_parents = [] # best parents in the current population

    while len(indices) >= k: # continue until there are enough remaining indices
        
        # remaining_indices = indices # each time used indices are removed, so these are the 'remaining' ones
        # remaining_population = [population[i] for i in remaining_indices] # remaining population according to indices
        # parents = random.choices(list(enumerate(remaining_population)), k=2) # choices will select k random values from population to compare
        # current_parents_indices = [index for index, _ in enumerate(inner_list[1::2] for inner_list in parents)] # will save th indices of the parents
        # # inner_list[1::2] for inner_list in parents, it selects every odd elemnt of the inner list, taking thus the indexes
        # current_fitness_scores = [fitness_score_array[i] for i in current_parents_indices] # fitness scores of the selected parents

        # print(current_parents_indices)
        # sorted_parents_indices = sorted(range(len(current_parents_indices)), key=lambda i: current_fitness_scores[i])
        # sorted_parents = [parent for _, parent in sorted(zip(current_fitness_scores, [value for value, _ in enumerate(inner_list[2::2] for inner_list in parents)]), key=lambda x: x[0])]
        # print('sorted parents'+str(sorted_parents))

        parents_indices = random.sample(indices, k=2)  # select two unique random indices (if choices used then the parents indices may be [3,3], allowing repetitions)
        parents = [population[i] for i in parents_indices]  # Get corresponding parents

        # select fitness scores for the selected parents according to their indices
        fitness_scores = [fitness_score_array[i] for i in parents_indices]

        # sort parents based on fitness scores (search for min)
        sorted_parents = [parent for _, parent in sorted(zip(fitness_scores, parents), key=lambda x: x[0])] # key is the first value of the sorted parents [0]

        best_parents.append(sorted_parents[0])

        # remove the index of the selected parent from the list of remaining indices
        for index in parents_indices:
            indices.remove(index)
        
        # remove the fitness scores of the unselected parents
        # for index in indices:
        #     fitness_score_array.remove(fitness_score_array[index])

    return best_parents


def single_point_crossover(best_parents: list):
    i=0
    # check for same length
    while i < len(best_parents) - 1:
        if len(best_parents[i]) != len(best_parents[i+1]):
            raise ValueError("Chromosomes must be of the same length")
        i+=1
    
    if len(best_parents) < 2:
        return best_parents

    crossovered_parents = []

    for i in range(0, len(best_parents) - 1, 2): # iterate through all unique pairs of 2 parents
        parent1 = best_parents[i]
        parent2 = best_parents[i+1]
        p = random.randint(1, len(parent1)) # generate a random integer within the bits length
        crossovered_parents.append(parent1[0:p]+parent2[p:]) # crossover a
        crossovered_parents.append(parent2[0:p]+parent1[p:]) # crossover b

    return crossovered_parents


def mutate(chromosomes: list, mutations_number, probability: float):

    indices = [i for i in range(len(chromosomes))] # will save th indices of the chromosomes
    # select a random chromosome to mutate from the given list
    chromosome_index = random.randint(0, len(chromosomes)-1)
    print(chromosome_index)
    chromosome = chromosomes[chromosome_index]
    chromosome = list(chromosomes[chromosome_index])  # convert string to list for mutability

    for _ in range(mutations_number):
        index = random.randrange(len(chromosome)) # pick a random bit (index) from the given chromosome
        if index in indices: # check if the index has already been mutated
            print('chromoIndex '+str(index))
            # random generates a random float within [0,1]
            # abs(chromosome[index] - 1) flips the bit from 0 to 1 and from 1 to 0
            chromosome[index] = chromosome[index] if random.random() > probability else abs(int(chromosome[index]) - 1)
            indices.remove(index)
        else:
            break
    
    # convert list back to string
    mutated_chromosome = ''.join(map(str, chromosome))
    chromosomes[chromosome_index] = mutated_chromosome  # update the original list

    return chromosomes


# elite_size refers to the # of best individuals to keep from the previous population
def replace_population(population: list, mutated_chromosomes: list, elite_size: int, fitness_score_array: list, mutated_fitness_score_array: list):

    # combine the current population & their scores with the mutated individuals
    combined_population = population + mutated_chromosomes
    combined_fitness_scores = fitness_score_array + mutated_fitness_score_array

    # sort based on fitness scores (ascending order)
    sorted_population = sorted(zip(combined_fitness_scores, combined_population), key=lambda x: x[0])

    # select as many best individuals as the # of th elite size
    new_population = [individual for _, individual in sorted_population[:elite_size]]

    return new_population


def calculate_chromatic_number(colors_matrix: list):

    unique_colors = set(color for row in colors_matrix for color in row)
    chromatic_number = len(unique_colors)
    return chromatic_number


def calculate_max_degree(adjacency_matrix: list):
    
    maximum_degree = max(sum(row) for row in adjacency_matrix)
    return maximum_degree


def main():

    # represent the graph through an adjacency matrix
    adjacency_matrix_file = 'adjacency_matrix.txt'
    adjacency_matrix = []

    if os.path.exists(adjacency_matrix_file):
        with open(adjacency_matrix_file, 'r') as file:
            for line in file:
                row = list(map(int, line.strip().split()))
                adjacency_matrix.append(row)
        print("Adjacency matrix loaded successfully:")
    else:
        print(f"Error: The file '{adjacency_matrix_file}' does not exist.")

    np_matrix = np.array(adjacency_matrix)
    print(np_matrix)
    rows, cols = np.where(np_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    mylabels = {node: f'{node + 1}' for node in G.nodes()}
    nx.draw(G, node_size=500, labels=mylabels ,with_labels=True)
    plt.show()

    
    # get number of colors
    num_colors = get_valid_number("Please enter a positive integer representing the number of colors: ")

    # power is # of bits needed to represent the colors (in this case 4)
    power, bits = smallest_power_of_2(num_colors)

    # get population
    P = get_valid_number("Please enter a positive integer representing the initial population: ")
    while P < int(np_matrix.shape[1]/2) + 1:
        P = get_valid_number("Initial population should be >= [(# of nodes)/2] + 1: ")

    population = generate_population(size=P, power=bits, dimensions=np_matrix.shape[1])
    print('initial population '+str(population)) 
    
    generation = 1 # 1st generation (1st parents)

    probability = get_probability("Please enter the mutation probability: ")
    mutations_number = get_valid_number("Please enter the number of mutations to perform on a single chromosome: ")
    max_generations = get_valid_number("Please enter the number of maximum generations that the algorithm will run: ")
    elite_size = get_valid_number("Please enter the number of the individuals from the new population to keep after the first generation of the algorithm comes to an end (the number is reduced by 10% of its value after each generation comes to an end): ")

    for generation in range(max_generations):

        print('current generation: '+str(generation))
        print('remaining population '+str(population))

        # evaluate current population
        fitness_score_array, colors_matrix = fitness(power=bits, population=population, adjacency_matrix=np_matrix)
        print('fitness scores '+str(fitness_score_array))

        # select best parents
        best_parents = tournament_selection(population=population, fitness_score_array=fitness_score_array, k=2)
        print("bestParents "+str(best_parents))

        #crossover
        crossover = single_point_crossover(best_parents=best_parents)
        print('crossovered '+str(crossover))

        # mutate crossovered children
        mutated_chromosomes = mutate(chromosomes=crossover, mutations_number=mutations_number, 
                                     probability=probability)
        print(mutated_chromosomes)

        # evaluate mutated individuals
        mutated_fitness_score_array, mutated_colors_matrix = fitness(power=bits, population=mutated_chromosomes, adjacency_matrix=np_matrix)
        print('mutated fitness scores '+str(mutated_fitness_score_array))

        chromatic_number = calculate_chromatic_number(colors_matrix=colors_matrix)
        maximum_degree = calculate_max_degree(adjacency_matrix=np_matrix)

        # check if the χ(G) ≤ Δ(G)+1 condition is satisfied
        # if chromatic_number <= maximum_degree + 1:
        #     print('chromatic number '+str(chromatic_number)+' , maximum degree '+str(maximum_degree))
        #     print("Condition satisfied. Stopping the genetic algorithm.")
        #     break
        # else:
        #     print("Condition not satisfied. Continuing the genetic algorithm.")

        population = replace_population(population=population, mutated_chromosomes=mutated_chromosomes, 
                           elite_size=elite_size, fitness_score_array=fitness_score_array, 
                           mutated_fitness_score_array=mutated_fitness_score_array)
        
        if len(population) < elite_size:
            print('Population diminished to minimum')
            break
        
        colors = colors_matrix[0]
        node_colors = [colors[node] for node in G.nodes()]

        # Draw the graph with node colors and labels
        nx.draw(G, node_size=500, labels=mylabels, with_labels=True, node_color=node_colors)
        plt.title('Graph with Node Colors, Generation '+str(generation))
        plt.show()

        elite_size = elite_size - (int(0.1 * elite_size))
        generation += 1

    return fitness_score_array

if __name__ == '__main__':
    main()