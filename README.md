
The program is designed to solve the graph coloring problem, where 13 nodes of a graph must be colored using 4 colors, ensuring no adjacent nodes share the same color. The graph is represented by its adjacency matrix, stored in a .txt file, and loaded into the program.

The solution employs a genetic algorithm with an initial population of chromosomes representing color assignments. The algorithm evolves this population using selection, crossover, and mutation to minimize the fitness score, which represents the number of adjacent nodes with the same color. The goal is to find the optimal coloring configuration, minimizing conflicts between adjacent nodes.

Key functions in the program include:

	1.	generate_population: Creates an initial population of chromosomes.
	2.	fitness: Evaluates each chromosome based on how many adjacent nodes have the same color.
	3.	tournament_selection: Selects the best solutions from the population based on fitness scores.
	4.	single_point_crossover: Implements the crossover of parents to create new offspring.
	5.	mutate: Introduces random mutations in the chromosomes to enhance diversity.

The program is implemented in Python, with parameterization options for the number of colors, population size, mutation probability, and the number of generations. Visualizations of the graph are produced using the matplotlib library.

![image](https://github.com/ioannisCC/genetic-algorithm-graph-coloring/assets/98465741/a7cad6d9-96d3-469a-9034-df844bf164d0)
