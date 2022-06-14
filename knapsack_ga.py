from cProfile import label
import numpy
import pygad
import matplotlib.pyplot as plt
import pandas as pd
import threading
from pathlib import Path

num_solutions = 150     #number of initial population
num_generations = 500    #number of total generations
num_parents_mating = 20  #Number of solutions to be selected as parents.
max_val_gen = 25        #Stop criteria: alg stop when max_val_gen lastest best fitness are the same
number_of_rows = 10     #Number of displayed rows in result csv.

filepath = Path('./KNAPSACK_output/{num_solutions}_{num_generations}.csv'.format(num_solutions=num_solutions, num_generations=num_generations))
inputpath = Path('./KNAPSACK_input/input.txt')
average_fitness_per_gen = []
weights = []
values = []
capacity = 0
gene_length = 0
count = 0
max_val = 0

class PygadThread(threading.Thread):
    
    def __init__(self, app, ga_instance):
        super().__init__()
        self.ga_instance = ga_instance
        self.app = app

    def run(self):
        self.ga_instance.run()
        # self.ga_instance.plot_fitness(title="Plot for {num_solutions} solutions and {num_generations} generations".format(num_solutions = num_solutions, num_generations = num_generations))

class GA():

    old_best_sol_fitness = -1
    old_best_sol_idx = -1

    def start_ga(self, *args):
        pygadThread = PygadThread(self, self.ga_instance)
        pygadThread.start()

    def read_input(self, *args):
        f = open(inputpath, "r")
        data = list(map(lambda x: list(map(lambda i: int(i),x.split())), f.read().split('\n')))
        global capacity, gene_length, weights, values
        gene_length = data[0][0]
        capacity = data[0][1]
        for i in range(gene_length):
            weights.append(data[i+1][0])
            values.append(data[i+1][1])
        print(capacity)

    def initialize_population(self, *args):
        global capacity, gene_length, weights, values
        self.population_1D_vector = numpy.zeros(shape=(num_solutions, gene_length)) # Each solution is represented as a row in this array. If there are 5 rows, then there are 5 solutions.

        # Creating the initial population RANDOMLY as a set of 1D vectors.
        for solution_idx in range(num_solutions):
            initial_y_indices = numpy.random.random_integers(low=0, high=1, size=gene_length)
            self.population_1D_vector[solution_idx, :] = initial_y_indices

        print("Population 1D Vectors : ", self.population_1D_vector)

        self.ga_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    fitness_func=fitness,
                                    num_genes=gene_length,
                                    initial_population=self.population_1D_vector,
                                    mutation_percent_genes=0.7,
                                    mutation_type="random",
                                    mutation_num_genes=1,
                                    mutation_by_replacement=True,
                                    gene_space=[0,1],
                                    on_generation=on_gen_callback,
                                    on_stop=on_stop_callback,
                                    delay_after_gen=0.2,
                                    crossover_type="two_points",
                                    parent_selection_type="rank",
                                    save_best_solutions=True)

def fitness(solution_vector, solution_idx):
    global capacity, gene_length, weights, values

    sum_w = 0
    sum_v = 0
    for i in range(len(weights)):
        sum_w += solution_vector[i]*weights[i]
        sum_v += solution_vector[i]*values[i]
    if (sum_w <= capacity):
        return sum_v
    return 0

def on_gen_callback(ga_instance):
    global count, max_val
    print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    population = ga_instance.population
    average_fitness_per_gen.append(numpy.average([fitness(population[i],i) for i in range(len(population))]))
    if (count <= max_val_gen):
        if (ga_instance.best_solution()[1] > max_val):
            max_val = ga_instance.best_solution()[1]
            count = 0
        else:  
            count += 1
    else:
        return "stop"

def on_stop_callback(ga_instance, last_generation_fitness): 
    best_solutions = ga_instance.best_solutions
    best_solutions_fitness = ga_instance.best_solutions_fitness
    print("best_solutions_fitness", best_solutions_fitness)
    plt.plot(average_fitness_per_gen, label="Average fitness")
    plt.plot(best_solutions_fitness, label="Max fitness")
    plt.legend()
    plt.title('Fitness vs generation over {population} initial solutions (population)'.format(population=num_solutions))
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
    print(average_fitness_per_gen)
    for i in range(len(best_solutions)):
        best_solutions[i] = numpy.append([i+1],numpy.append(best_solutions[i],[best_solutions_fitness[i]]))
    columns = numpy.append(["Generation"],numpy.append(["Column {d}".format(d=i) for i in range(gene_length)], ['Fitness score']))
    df = pd.DataFrame(sorted(best_solutions, key=lambda x:x[0], reverse=True)[:number_of_rows], columns=columns)
    df.to_csv(path_or_buf=filepath, index=False)
app = GA()
app.read_input()
app.initialize_population()
app.start_ga()