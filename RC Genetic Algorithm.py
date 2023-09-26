import numpy as np
import random as rnd
import copy
import time
from matplotlib import pyplot as plt
import os
import operator
from tabulate import tabulate
import pandas as pd
import statistics

'''Parameters'''
population_size = 20; num_genes = 3; limits = [0., 10.]; # Problem Size and Boundaries
selection_method ='tournment'; tournment_players_num = 2; strong_to_win = 0.9; # Selection Parameters
crossover_method = 'BLX-a'; crossover_prop = 0.8; # Cross-Over parameters
mutation_method = 'random'; mutation_prop = 0.01; # Mutation Parameters
num_elites = 0; # Elistism (number of elite solutions)
max_gen = 100; tolerance = 1e-6; # Termination parameters
refreshment_flag = 0 # Population disrubtion flag
abs_path = os.path.abspath(__file__); results_dir = os.path.join(os.path.dirname(abs_path), 'Results/') # Path for saving Results.

class Chromosome:

    def __init__(self, genes = []):
        self.genes = genes[:]
        self.number_of_genes = len(genes)
        self.fitness = -1
        self.bounds = None
    
    def __fitness_function__(self, mode = 'minimize'):
        if self.genes is None:
            return -1
        x1 = self.genes[0]; x2 = self.genes[1]; x3 = self.genes[2]
        fitness = pow(x1,2) + 2*pow(x2,2) + 3*pow(x3,2) \
        + x1*x2 + x2*x3 - 8*x1 - 16*x2 - 32*x3 + 110

        if fitness == 0:
            fitness = 1e-20     # Case: division by Zero

        if mode == 'minimize':
            return 1/fitness
        else:
            return fitness

    def set_genes(self, genes):
        self.genes = genes[:]
        self.number_of_genes = len(self.genes)
        #self.fitness_eval()

    def fitness_eval(self, func = None):
        if(func == None):
            self.fitness = self.__fitness_function__()
        else:
            self.fitness = func()
        
        return self.fitness

    def print_chromosome(self):
        print(*self.genes)

class Genetic_Algorithm:

    def __init__(self):
        self.current_population = []
        self.new_population = []
        self.best_individual_idx = 0
        self.current_population_fitness = []
        self.current_fitness_ratios = []
        self.parent_pairs = []
        self.global_optimum = Chromosome()
        self.generation = 0
        self.max_generations = 0
        self.current_error = 0
        self.stuck_percent = 0
        self.number_of_refreshments = 0
        self.best_individual_history = []
        self.global_optimum_history = []
        self.current_error = 1.; self.last_error = 1.

    def __Tournment_Selection__(self, players, stgr_w_per = 0.9):
        '''1. Indices of individuals that will go for a tournment,
        2. Stronger to win percentage'''

        prop = np.random.uniform(0.0,1.0)
        if max(self.current_population_fitness) == min(self.current_population_fitness):
            return rnd.choice(players)

        if(prop < stgr_w_per):
           return players[np.argmax(np.take(self.current_population_fitness, players))]
        else:
            return players[np.argmin(np.take(self.current_population_fitness, players))] 
     
    def __Roullete_Wheel_Selection__(self):
        prop = np.random.uniform(0.0,1.0)
        if prop == 0:
            prop = 1 # Case: prop is the sarting point in the roullete wheel, go for the end point

        temp_sum = sum(self.current_population_fitness)
        fitness_ratios = [fitness/temp_sum \
                                           for fitness in self.current_population_fitness]
        wheel = [0]
        for idx,ratio in enumerate(fitness_ratios):
            wheel.append(wheel[idx] + ratio)
        
        for idx in range(len(wheel)-1):
            if prop < wheel[idx + 1] and prop > wheel[idx]:
                return idx

    def __Flat_CrossOver__(self, dads, moms):
        '''1. Array of Left hand side parents
        2. Array of Right hand side parents'''

        for i in range(len(dads)):
            genetics_1 = []; genetics_2 = []
            # Generate two children within the range of parents
            for j in range(dads[i].number_of_genes):
                new_gene_1, new_gene_2 = np.random.uniform(dads[i].genes[j],moms[i].genes[j],2)
                genetics_1.append(new_gene_1)
                genetics_2.append(new_gene_2)
                    
            self.new_population.append(Chromosome(genetics_1))
            self.new_population.append(Chromosome(genetics_2))
    
    def __Simple_CrossOver__(self, dads, moms):
        '''1. Array of Left hand side parents
        2. Array of Right hand side parents'''

        for i in range(len(dads)):
            rand_idx = rnd.randint(0,dads[0].number_of_genes-2)
            genetics_1 = dads[i].genes[:rand_idx+1] + moms[i].genes[rand_idx+1:]
            genetics_2 = moms[i].genes[:rand_idx+1] + dads[i].genes[rand_idx+1:]
            self.new_population.append(Chromosome(genetics_1))
            self.new_population.append(Chromosome(genetics_2))

    def __Arethmetical_CrossOver__(self, dads, moms):    
        for i in range(len(dads)):
            combination_factor = np.random.uniform(0.0,1.0)
            genetics_1 = list(np.add(combination_factor*(np.subtract(dads[i].genes,moms[i].genes)),moms[i].genes))
            genetics_2 = list(np.add(combination_factor*(np.subtract(moms[i].genes,dads[i].genes)),dads[i].genes))                   
            self.new_population.append(Chromosome(genetics_1))
            self.new_population.append(Chromosome(genetics_2))

    def __BLX_Alpha_CrossOver__(self, dads, moms):
        for i in range(len(dads)):
            genetics_1 = []; genetics_2 = []
            alpha = 0.1 #np.random.uniform(0.0,0.5)
            # Generate two children within the modified range of parents (Exploration + Exploitation)

            for j in range(dads[i].number_of_genes):
                c_max = max([dads[i].genes[j], moms[i].genes[j]])
                c_min = min([dads[i].genes[j], moms[i].genes[j]])
                h_max = c_max + alpha*(c_max - c_min)
                h_min = c_min - alpha*(c_max - c_min)
                new_gene_1, new_gene_2 = np.random.uniform(h_min,h_max,2)

                # Feasibility Assurance
                if not (self.bounds is None):
                    if new_gene_1 < min(self.bounds): new_gene_1 = min(self.bounds)
                    if new_gene_1 > max(self.bounds): new_gene_1 = max(self.bounds)
                    if new_gene_2 < min(self.bounds): new_gene_2 = min(self.bounds)
                    if new_gene_2 > max(self.bounds): new_gene_2 = max(self.bounds)

                genetics_1.append(new_gene_1)
                genetics_2.append(new_gene_2)
                    
            self.new_population.append(Chromosome(genetics_1))
            self.new_population.append(Chromosome(genetics_2))

    def __Linear_CrossOver__(self, dads, moms):
       for i in range(len(dads)):
            genetics_1 = []; genetics_2 = []; genetics_3 = []
            # Generate three children & choose the best 2
            for j in range(dads[i].number_of_genes):
                new_gene_1, new_gene_2, new_gene_3 = 0.5*(dads[i].genes[j] + moms[i].genes[j]),\
                3/2*dads[i].genes[j] - 0.5*moms[i].genes[j], 3/2*moms[i].genes[j] - 0.5*dads[i].genes[j]

                # Feasibility Assurance
                if not (self.bounds is None):
                    if new_gene_1 > max(self.bounds): new_gene_1 = max(self.bounds)
                    if new_gene_2 < min(self.bounds): new_gene_2 = min(self.bounds)
                    if new_gene_2 > max(self.bounds): new_gene_2 = max(self.bounds)
                    if new_gene_3 < min(self.bounds): new_gene_3 = min(self.bounds)
                    if new_gene_3 > max(self.bounds): new_gene_3 = max(self.bounds)

                genetics_1.append(new_gene_1)
                genetics_2.append(new_gene_2)
                genetics_3.append(new_gene_3)

            genetics_list = [genetics_1, genetics_2, genetics_3]
            fitness_list = [Chromosome(genetics_1).fitness_eval(), Chromosome(genetics_2).fitness_eval(),\
                                        Chromosome(genetics_3).fitness_eval()]
            genetics_list.pop(fitness_list.index(min(fitness_list))) # Drop the least fit Genetic pattern 

            self.new_population.append(Chromosome(genetics_list[0]))
            self.new_population.append(Chromosome(genetics_list[1]))

    def __Discrete_CrossOver__(self,dads,moms):
        for i in range(len(dads)):
            genetics_1 = []; genetics_2 = []
            # Generate two children of genes chosen discretely from the parent's pole
            for j in range(dads[i].number_of_genes):
                new_gene_1, new_gene_2 = np.random.choice([dads[i].genes[j], moms[i].genes[j]], size = 2)
                genetics_1.append(new_gene_1)
                genetics_2.append(new_gene_2)
                    
            self.new_population.append(Chromosome(genetics_1))
            self.new_population.append(Chromosome(genetics_2))

    def __Random_Mutation__(self):

        i = rnd.randint(0,self.population_size-1)
        j = rnd.randint(0,self.new_population[i].number_of_genes-1)
        self.new_population[i].genes[j] = np.random.uniform(self.bounds[0],self.bounds[1])
    
    def __Non_Uniform_Mutation__(self, b = 2):
        '''1. degree of dependency on the number of generations'''

        i = rnd.randint(0,self.population_size-1)
        j = rnd.randint(0,self.new_population[i].number_of_genes-1)
        taw = np.random.choice([0,1], size = 1)
        c = self.new_population[i].genes[j]

        if taw == 0:
            y = max(self.bounds) - c
        else:
            y = -1*(c - min(self.bounds))

        r = np.random.uniform(0.0,1.0)
        del_fact = y * (1 - pow(r,1-pow(self.generation/self.max_generations,b)))
        c_dash = c + del_fact

        # Feasibility Assurance
        if c_dash > max(self.bounds): c_dash = max(self.bounds)
        if c_dash < min(self.bounds): c_dash = min(self.bounds)
        self.new_population[i].genes[j] = c_dash

    def Initialize_GA(self, pop_size, n_o_g, bounds = None, max_gen = 500):
        '''Creates an initial population'''
        
        self.bounds = bounds[:]
        self.population_size = pop_size
        self.current_population = [Chromosome([np.random.uniform(self.bounds[0],self.bounds[1]) for i in range(n_o_g)])\
                                   for j in range(self.population_size)]
        self.generation = 1
        self.max_generations = max_gen
        
    def Evaluate_Population_Fitness(self):
        '''Evaluates the population's Fitness'''

        self.current_population_fitness = [chromosome.fitness_eval() for chromosome in self.current_population]
        self.best_individual_idx = self.current_population_fitness.index(max(self.current_population_fitness))
        self.best_individual_history.append(self.current_population_fitness[self.best_individual_idx])
        

    def Update_Global_Optimum(self):
        '''Updates Global Optimum'''
        
        if self.current_population_fitness[self.best_individual_idx] > self.global_optimum.fitness:
            self.global_optimum = copy.deepcopy(self.current_population[self.best_individual_idx])
        else:
            self.stuck_percent = ((self.stuck_percent*self.max_generations) + 1)/self.max_generations
        
        self.last_error = self.current_error
        self.current_error = abs(self.global_optimum.fitness - self.current_population_fitness[self.best_individual_idx])/\
            self.global_optimum.fitness
        
        self.global_optimum_history.append(self.global_optimum.fitness)


    
    def Select_Parents(self, method = 'tournment', tournment_players_num = 2, stronger_win_prob = 0.9):
        '''The driver function for selection:
        1. Selection method,
        2. tournment players in case tournment method chosen'''

        if method == 'roullete':
            for i in range(0,self.population_size, 2):
                dad_idx = self.__Roullete_Wheel_Selection__()
                mom_idx = self.__Roullete_Wheel_Selection__()
                
                while mom_idx == dad_idx or (mom_idx, dad_idx) in self.parent_pairs or (dad_idx,mom_idx) in self.parent_pairs:
                    dad_idx = self.__Roullete_Wheel_Selection__()
                    mom_idx = self.__Roullete_Wheel_Selection__()

                self.parent_pairs.append((mom_idx, dad_idx))

        elif method == 'tournment':
            tournment_players = list(range(0,self.population_size))
            for i in range(0,self.population_size, 2):

                dad_players = rnd.sample(tournment_players,tournment_players_num)
                dad_idx = self.__Tournment_Selection__(players=dad_players, stgr_w_per=stronger_win_prob)

                mom_players = rnd.sample(tournment_players,tournment_players_num)
                mom_idx = self.__Tournment_Selection__(players=mom_players, stgr_w_per=stronger_win_prob)

                while (mom_idx, dad_idx) in self.parent_pairs or (dad_idx,mom_idx) in self.parent_pairs:
                    dad_players = rnd.sample(tournment_players,tournment_players_num)
                    dad_idx = self.__Tournment_Selection__(players=dad_players, stgr_w_per=stronger_win_prob)

                    mom_players = rnd.sample(tournment_players,tournment_players_num)
                    mom_idx = self.__Tournment_Selection__(players=mom_players, stgr_w_per=stronger_win_prob)
                
                self.parent_pairs.append((mom_idx, dad_idx))

    def Cross_Over(self,method = 'flat', cross_prop = 0.8):
        '''The driver function for the cross-over:
        1.Cross-Over method
        2. Cross-Over probability'''

        prop = np.random.uniform(0.0,1.0)
        unzipped_pairs_list = list(zip(*self.parent_pairs))
        dads = np.take(self.current_population, unzipped_pairs_list[0][:]).tolist()
        moms = np.take(self.current_population, unzipped_pairs_list[1][:]).tolist()

        if prop > cross_prop:
            self.new_population.extend(dads + moms)
            
        else:
            if method == 'flat':
                self.__Flat_CrossOver__(dads, moms)
                
            elif method == 'simple':
                self.__Simple_CrossOver__(dads,moms)

            elif method == 'arithmetical':
                self.__Arethmetical_CrossOver__(dads,moms)
            
            elif method == 'BLX-a':
                self.__BLX_Alpha_CrossOver__(dads,moms)

            elif method == 'linear':
                self.__Linear_CrossOver__(dads,moms)
            
            elif method == 'discrete':
                self.__Discrete_CrossOver__(dads,moms)

            else:
                self.__Simple_CrossOver__(dads,moms)

            # Set Flag to 1 to debug the Cross-Over
            Flag = 0
            if Flag:
                for i in range(len(dads)):
                    dads[i].print_chromosome()
                    moms[i].print_chromosome()
                    self.new_population[2*i].print_chromosome()
                    self.new_population[2*i+1].print_chromosome()

                    print()
                print(i)

    def Mutation(self, method = 'random', mutat_prop = 0.01):
        '''The driver function for the Mutation:
        1. Mutation method
        2. Mutation probability'''

        prop = np.random.uniform(0.0,1.0)
        if prop < mutat_prop:
            if method == 'random':
                self.__Random_Mutation__()
            elif method == 'non-uniform':
                self.__Non_Uniform_Mutation__()
            else: None
        
        Flag = 0
        if Flag:
            for i in range(len(self.new_population)):
                print(*self.new_population[i].genes)

    def Pass_Elite(self, elite_size = 1):
        ''' 1. Number of elites'''
        if elite_size > self.population_size: elite_size = self.population_size
        elif elite_size <= 0: return
        else:
            zipped_elites = sorted(zip(self.current_population_fitness, self.current_population), \
                                   key=operator.itemgetter(0), reverse=True)
            unzipped_elites = list(zip(*zipped_elites))
            elites = unzipped_elites[1][:]
            elites = list(elites[:elite_size])

            # Replace solutions (selected randomly, could be optimised to replace the worst)
            self.new_population[:elite_size] = elites
       

    def __Refresh_GA__(self):
        self.current_population[round(self.population_size/2):] = \
            [Chromosome([np.random.uniform(self.bounds[0],self.bounds[1]) \
                         for i in range(self.current_population[0].number_of_genes)])\
                                   for j in range(self.population_size)]
        
    def __finalise_new_population__(self, refreshment_flag = 0):
        
        if refreshment_flag and self.stuck_percent >= 0.2:
            self.__Refresh_GA__()
            self.stuck_percent = 0
            self.number_of_refreshments = self.number_of_refreshments + 1

        else:
            if len(self.new_population) > self.population_size:
                self.new_population = self.new_population[len(self.new_population) - self.population_size:]
                # Could be optimised to remove the worst solutions: Future Implementation
            elif len(self.new_population) < self.population_size:
                extension = [Chromosome([np.random.uniform(self.bounds[0],self.bounds[1])\
                                          for i in range(self.current_population[0].number_of_genes)])\
                                            for j in range(self.population_size - len(self.new_population))]
                self.new_population.extend(extension)
        
            self.current_population = self.new_population[:]
        self.new_population.clear()   
        self.parent_pairs.clear()


    def __Run_Algorithm__(self):
        
        self.Initialize_GA(pop_size = population_size, n_o_g = num_genes, bounds = limits, max_gen = max_gen)
        while(self.generation < max_gen):

            self.Evaluate_Population_Fitness()
            #self.print_population(self.current_population)
            if self.generation > 1:
                # If the deviation between local and global remains constant, and the global doesn't change
                if  abs(self.current_error - self.last_error) < tolerance and self.stuck_percent > 0.2:
                    self.generation = self.generation + 1
                    self.Update_Global_Optimum()
                    return self.global_optimum
            
            self.Update_Global_Optimum()
        
            self.Select_Parents(method = selection_method,\
                                tournment_players_num=tournment_players_num, stronger_win_prob=strong_to_win)
            self.Cross_Over(method=crossover_method,cross_prop=crossover_prop)
            self.Mutation(method=mutation_method,mutat_prop=mutation_prop)
            self.Pass_Elite(elite_size= num_elites)
            #self.print_population(self.current_population)
            self.__finalise_new_population__(refreshment_flag)
            
            self.generation = self.generation + 1
            #print("Generation: ", self.generation)

        return self.global_optimum

    def Run(self):
        self.__Run_Algorithm__()
    
    def print_population(self, population):
        for i in range(len(population)):
            population[i].print_chromosome()
            print('Fitness: ', population[i].fitness)
        
        print()

def report_generator(param):
    global strong_to_win, crossover_prop, mutation_prop, num_elites, results_dir

    results_dir = os.path.join(os.path.dirname(abs_path), 'Results/%s/' %param)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    repitition = 10
    opt_locals = []; opt_globals = []; opt_sols = []
    if param == "stbs" or param == "x-over":
        parameters = np.linspace(0.5,0.9,10)
    elif param == "mutation":
        parameters = np.linspace(0.00,0.05,6)
    elif param == "elitism":
        parameters = np.linspace(0,10,6)
    else: parameters = []
        
    runs = list(range(1, repitition + 1))

    for i in range(len(parameters)):

        if param == "stbs":
            strong_to_win = round(parameters[i],3)
            printer = 'Stronger to win probability = ' + str(strong_to_win)
            printer2 = 'Stronger to win probability'

        elif param == "x-over":
            crossover_prop = round(parameters[i],3)
            printer = 'Cross-Over probability = ' + str(crossover_prop)
            printer2 = 'Cross-Over probability'

        elif param == "mutation":
            mutation_prop = round(parameters[i],4)
            printer = 'Mutation probability = ' + str(mutation_prop)
            printer2 = 'Mutation probability'

        elif param == "elitism":
            num_elites = round(parameters[i])
            printer = 'Elites Percentage = ' + str(num_elites/population_size*100) + '%'
            printer2 = 'Elites Percentage'

        else:
            printer = None
            printer2 = None

        sols = []; local_histories = []; global_histories = []; iterations_list = []; global_sols = []
        for j in range(repitition):
            problem = Genetic_Algorithm()
            problem.Run()
            sols.append(problem.global_optimum.fitness)
            local_histories.append(problem.best_individual_history)
            global_histories.append(problem.global_optimum_history)
            global_sols.append(problem.global_optimum.genes)
            iterations = list(range(1, problem.generation))
            iterations_list.append(iterations)

        idx_plot = sols.index(max(sols))
        #opt_locals.append(local_histories[idx_plot][-1])
        opt_globals.append(global_histories[idx_plot][-1])
        opt_sols.append(global_sols[idx_plot])

        plt.figure(1)
        plt.plot(iterations_list[idx_plot], local_histories[idx_plot],label='Local Optimum Fitness')
        plt.plot(iterations_list[idx_plot], global_histories[idx_plot],'-.', label='Global Optimum Fitness')
        plt.xlabel('Generation'); plt.ylabel('Fitness')
        plt.legend()
        plt.grid()
        plt.title(printer)
        #plt.show() 
        plt.savefig(results_dir + "LocalGlobal_%s.png"  %printer)
        plt.clf()

        plt.figure(2)
        #local_finals = [local_histories[i][-1] for i in range(repitition)]
        global_finals = [global_histories[i][-1] for i in range(repitition)]

        #plt.plot(runs, local_finals, label = 'Final Local Fitness')
        plt.plot(runs, global_finals,'-o', label = 'Final Global Fitness')
        plt.xlabel('Run'); plt.ylabel('Final Soultion Fitness')
        plt.legend()
        plt.grid()
        plt.title(printer)
        #plt.show() 
        plt.savefig(results_dir + "trials_%s.png"  %printer)
        plt.clf()

    plt.figure(3)
    #plt.plot(parameters, opt_locals,'-o', label = 'Optimum Final Local Fitness')
    plt.plot(parameters, opt_globals,'-o', label = 'Optimum Global Fitness')
    plt.xlabel(printer2); plt.ylabel('Final Soultion Fitness')
    plt.legend()
    plt.grid()
    plt.title('Effect of ' + printer2)
    #plt.show()
    plt.savefig(results_dir + "output_%s.png" %printer2)

    opt_sols = np.array(opt_sols)
    table = [parameters]; table.extend([opt_sols[:,i] for i in range(len(opt_sols[0,:]))]); table.append(opt_globals)
    df = pd.DataFrame(table).T
    #heads = [printer2,'x_1','x_2','x_3', 'Fitness']
    #df.columns= heads
    print(df)
    df.to_csv(results_dir + 'output_%s.csv' %printer2, index=False, header=False)


def refreshment_test():
    global strong_to_win, crossover_prop, mutation_prop, num_elites, results_dir, refreshment_flag

    results_dir = os.path.join(os.path.dirname(abs_path), 'Results/refreshment/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    refreshment_flags = [0,1]
    combinations  = 5
    repitition = 10
    runs = list(range(1, repitition + 1))
    
    ref_noref_avergae = []; strong_to_win_list = []
    crossover_prop_list = []; mutation_prop_list = []

    for combination in range(combinations):
        strong_to_win_list.append(np.random.uniform(0.7,0.9))
        crossover_prop_list.append(np.random.uniform(0.7,0.9))
        mutation_prop_list.append(np.random.uniform(0.01,0.05))
    
    df = pd.DataFrame([strong_to_win_list, crossover_prop_list, mutation_prop_list]).T
    df.to_csv(results_dir + 'hyper_ref.csv', index=False, header=['STBS', 'Cross-Over', 'Mutation'])

    for refreshment_flag in refreshment_flags:
        average_globals = []; nos_refresh = []
        for combination in range(combinations):
            strong_to_win = strong_to_win_list[combination]
            crossover_prop = crossover_prop_list[combination]
            mutation_prop = mutation_prop_list[combination]

            sols = []; local_histories = []; global_histories = []; iterations_list = []; global_sols = []
            refreshes = []
            for run in range(repitition):
                
                problem = Genetic_Algorithm()
                problem.Run()
                sols.append(problem.global_optimum)
                local_histories.append(problem.best_individual_history)
                global_histories.append(problem.global_optimum_history)
                global_sols.append(problem.global_optimum.genes)
                iterations = list(range(1, problem.generation))
                iterations_list.append(iterations)
                if refreshment_flag:
                    refreshes.append(problem.number_of_refreshments)

                if 0:
                    print("Number of Generations: ", problem.generation)
                    print("Optimum Solution: ", *problem.global_optimum.genes, ", Fitness: ", problem.global_optimum.fitness)
                    print("Stuck Percent: ", problem.stuck_percent)
                    print("Number of Refreshments: ", problem.number_of_refreshments)

            if refreshment_flag:
                nos_refresh.append(refreshes)

            plt.figure(1)
            #local_finals = [local_histories[i][-1] for i in range(repitition)]
            global_finals = [global_histories[i][-1] for i in range(repitition)]

            #plt.plot(runs, local_finals, label = 'Final Local Fitness')
            if refreshment_flag:
                printer = 'With Refreshment'
            else:
                printer = 'Without Refreshment'
            plt.plot(runs, global_finals,'-o', label = printer)
            plt.xlabel('Run'); plt.ylabel('Final Soultion Fitness')
            plt.legend()
            plt.grid()
            plt.title("STBS, Cross-Over, Mutation Probabilities = %s" \
                      %str([round(strong_to_win,3), round(crossover_prop,3), round(mutation_prop,4)]))
            #plt.show() 
            plt.savefig(results_dir + "trials_ref%s.png"  %[refreshment_flag, combination])
            plt.clf()

            average_globals.append(statistics.mean(global_finals))

        #print(average_globals)
        ref_noref_avergae.append(average_globals)
    
    plt.figure(2)

    plt.plot(list(range(1,combinations+1)), ref_noref_avergae[0],'-o', label = 'No Refreshment')
    plt.plot(list(range(1,combinations+1)), ref_noref_avergae[1],'-o', label = 'With Refreshment')
    plt.xlabel('Combination'); plt.ylabel('Average Solution Fitness')
    plt.legend()
    plt.grid()
    plt.title("Average Fitness of Each Combination" )
    #plt.show() 
    plt.savefig(results_dir + "avg_ref%s.png")

    nos_refresh = np.array(nos_refresh)
    df = pd.DataFrame(nos_refresh).T
    print(df)
    df.to_csv(results_dir + 'output_ref.csv', index=False, header=False)

def basic_runs():
    repetitions = 9
    global strong_to_win, crossover_prop, mutation_prop, results_dir

    results_dir = os.path.join(os.path.dirname(abs_path), 'Results/basic/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #first_run_params = [0.9, 0.8, 0.01]
    strong_to_win, crossover_prop, mutation_prop = 0.9, 0.8, 0.01
    params = [[strong_to_win, crossover_prop, mutation_prop]]

    problem = Genetic_Algorithm()
    problem.Run()
    printer = "strong_to_win, crossover_prop, mutation_prop = 0.9, 0.8, 0.01"
    plt.figure(1)
    plt.plot(list(range(1,problem.generation)), problem.best_individual_history, label='Local Optimum Fitness')
    plt.plot(list(range(1,problem.generation)), problem.global_optimum_history, label='Local Optimum Fitness')
    plt.xlabel('Generation'); plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.title(printer)
    #plt.show() 
    plt.savefig(results_dir + "LocalGlobal_0.png")
    plt.clf()

    sols = [problem.global_optimum.genes]
    fitn = [problem.global_optimum.fitness]

    for i in range(repetitions):
        #first_run_params = [0.9, 0.8, 0.01]
        strong_to_win, crossover_prop, mutation_prop = np.random.uniform(0.7,0.9),\
              np.random.uniform(0.7,0.9), np.random.uniform(0.01,0.05)
        
        params.append([strong_to_win, crossover_prop, mutation_prop])

        problem = Genetic_Algorithm()
        problem.Run()
        printer = "strong_to_win, crossover_prop, mutation_prop = %.3f, %.3f, %.4f" %(strong_to_win,crossover_prop,mutation_prop)
        plt.figure(1)
        plt.plot(list(range(1,problem.generation)), problem.best_individual_history, label='Local Optimum Fitness')
        plt.plot(list(range(1,problem.generation)), problem.global_optimum_history, label='Global Optimum Fitness')
        plt.xlabel('Generation'); plt.ylabel('Fitness')
        plt.legend()
        plt.grid()
        plt.title(printer)
        #plt.show() 
        plt.savefig(results_dir + "LocalGlobal_%s.png"  %(i+1))
        plt.clf()
        sols.append(problem.global_optimum.genes)
        fitn.append(problem.global_optimum.fitness)
    params = np.array(params)
    sols = np.array(sols)
    table = [params[:,i] for i in range(len(params[0,:]))]; table.extend([sols[:,i] for i in range(len(sols[0,:]))]);
    table.append(fitn)
    df = pd.DataFrame(table).T
    print(df)
    df.to_csv(results_dir + 'output_basic.csv', index=False, header=False)



if __name__ == "__main__":
    rnd.seed(time.time())

    if 1: 
        if 1:
            report_generator(param='stbs')
        elif 0:
            problem = Genetic_Algorithm()
            problem.Run()
            if 1:
                print("Number of Generations: ", problem.generation)
                print("Optimum Solution: ", *problem.global_optimum.genes, ", Fitness: ", problem.global_optimum.fitness)
                print("Stuck Percent: ", problem.stuck_percent)
                print("Number of Refreshments: ", problem.number_of_refreshments)
        else:
            refreshment_test()
    else:
        basic_runs()
