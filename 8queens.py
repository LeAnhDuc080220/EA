import numpy
import pygad
import matplotlib.pyplot as plt
import pandas as pd
import threading
from pathlib import Path

num_solutions = 150     #number of initial population
num_generations = 500    #number of total generations
num_parents_mating = 20  #Number of solutions to be selected as parents.
number_of_rows = 10     #Number of displayed rows in result csv.

filepath = Path('./{num_solutions}_{num_generations}.csv'.format(num_solutions=num_solutions, num_generations=num_generations))

average_fitness_per_gen = []

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

    def initialize_population(self, *args):
        self.population_1D_vector = numpy.zeros(shape=(num_solutions, 8)) # Each solution is represented as a row in this array. If there are 5 rows, then there are 5 solutions.

        # Creating the initial population RANDOMLY as a set of 1D vectors.
        for solution_idx in range(num_solutions):
            initial_queens_y_indices = numpy.random.random_integers(low=0, high=7, size=8)
            self.population_1D_vector[solution_idx, :] = initial_queens_y_indices

        print("Population 1D Vectors : ", self.population_1D_vector)

        self.ga_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    fitness_func=fitness,
                                    num_genes=8,
                                    initial_population=self.population_1D_vector,
                                    mutation_percent_genes=0.8,
                                    mutation_type="random",
                                    mutation_num_genes=1,
                                    mutation_by_replacement=True,
                                    gene_space=[0,1,2,3,4,5,6,7],
                                    on_generation=on_gen_callback,
                                    on_stop=on_stop_callback,
                                    delay_after_gen=0.2,
                                    crossover_type="two_points",
                                    parent_selection_type="rank",
                                    save_best_solutions=True)

def fitness(solution_vector, solution_idx):
    solution = [round(x) - 1 if round(x) == 8 else round(x) for x in solution_vector]

    total_num_attacks = attacks(solution)

    if (total_num_attacks == 0):
        return 1.1
    else:
        return 1/total_num_attacks

def attacks(ga_solution):
    # For a given queen, how many attacking queen pairs? This is how the fitness value is calculated.

    total_num_attacks = 0 # Number of attacks for the solution.
        
    for i in range(7):
        for j in range(i+1, 8):
            total_num_attacks += (ga_solution[i] == ga_solution[j]) | (abs(i-j) == abs(ga_solution[i] - ga_solution[j]))

    return total_num_attacks

def on_gen_callback(ga_instance):
    print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Population = {population}".format(population=len(ga_instance.population)))
    population = ga_instance.population
    average_fitness_per_gen.append(numpy.average([fitness(population[i],i) for i in range(len(population))]))

    if (ga_instance.best_solution()[1] == 1.1):
        return "stop"

def on_stop_callback(ga_instance, last_generation_fitness): 
    best_solutions = ga_instance.best_solutions
    plt.plot(average_fitness_per_gen)
    plt.title('Average fitness vs generation over {population} initial solutions (population)'.format(population=num_solutions))
    plt.xlabel('Generations')
    plt.ylabel('Average fitness')
    plt.show()
    print(average_fitness_per_gen)
    best_solutions_fitness = ga_instance.best_solutions_fitness
    for i in range(len(best_solutions)):
        best_solutions[i] = numpy.append([i+1],numpy.append(best_solutions[i],[best_solutions_fitness[i]]))
        print(i)
    columns = numpy.append(["Generation"],numpy.append(["Column {d}".format(d=i) for i in range(8)], ['Fitness score']))
    df = pd.DataFrame(sorted(best_solutions, key=lambda x:x[0], reverse=True)[:number_of_rows], columns=columns)
    df.to_csv(path_or_buf=filepath, index=False)
app = GA()
app.initialize_population()
app.start_ga()