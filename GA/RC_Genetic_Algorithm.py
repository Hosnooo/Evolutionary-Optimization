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
import os
import logging
import math

'''Parameters'''
population_size = 50; num_genes = 3; limits = [[0,0,0],[10,10,10]]; # Problem Size and Boundaries
selection_method ='tournment'; tournment_players_num = 2; strong_to_win = 0.9; # Selection Parameters
crossover_method = 'BLX-a'; crossover_prop = 0.8; # Cross-Over parameters
mutation_method = 'random'; mutation_prop = 0.01; # Mutation Parameters
num_elites = 0; # Elistism (number of elite solutions)
max_gen = 100; tolerance = 1e-6; # Termination parameters
refreshment_flag = 0 # Population disrubtion flag

total_evals = 0 # Evaluations Count

class Chromosome:

    def __init__(self, genes = []):
        self.genes = genes[:]
        self.number_of_genes = len(genes)
        self.fitness = -1
    
    def __fitness_function__(self):
        if self.genes is None:
            return -1
        x1 = self.genes[0]; x2 = self.genes[1]; x3 = self.genes[2]
        cost = pow(x1,2) + 2*pow(x2,2) + 3*pow(x3,2) \
        + x1*x2 + x2*x3 - 8*x1 - 16*x2 - 32*x3 + 110
        # fitness = self.genes[0]*(1+self.genes[1]) - 1e3*(abs(pow(self.genes[0],2) + pow(self.genes[1],2) - 1))
        return cost

    def set_genes(self, genes):
        self.genes = genes[:]
        self.number_of_genes = len(self.genes)
        #self.fitness_eval(self.obj_func)

    def fitness_eval(self, func = None):
        global total_evals
        total_evals = total_evals + 1

        if(func == None):         
            cost = self.__fitness_function__()
        else:
            cost = func(self.genes)

        if cost == 0:
            cost = 1e-20     # Case: division by Zero
        
        self.fitness = 1/cost


        
        return self.fitness

    def print_chromosome(self):
        print(*self.genes)

class Genetic_Algorithm:

    def __init__(self, obj_func = None, bounds = limits):
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
        self.obj_func = obj_func
        self.bounds = bounds[:]

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
    
    def __Simple_CrossOver__(self, dad, mom):
        '''1. Left hand side parent
        2. Right hand side parent'''

        rand_idx = rnd.randint(0,dad.number_of_genes-2)
        genetics_1 = dad.genes[:rand_idx+1] + mom.genes[rand_idx+1:]
        genetics_2 = mom.genes[:rand_idx+1] + dad.genes[rand_idx+1:]
        self.new_population.append(Chromosome(genetics_1))
        self.new_population.append(Chromosome(genetics_2))

    def __Arethmetical_CrossOver__(self, dad, mom):
        '''1. Left hand side parent
        2. Right hand side parent'''

        combination_factor = np.random.uniform(0.0,1.0)
        genetics_1 = list(np.add(combination_factor*(np.subtract(dad.genes,mom.genes)),mom.genes))
        genetics_2 = list(np.add(combination_factor*(np.subtract(mom.genes,dad.genes)),dad.genes))                   
        self.new_population.append(Chromosome(genetics_1))
        self.new_population.append(Chromosome(genetics_2))

    def __BLX_Alpha_CrossOver__(self, dad, mom):
        '''1. Left hand side parent
        2. Right hand side parent'''

        genetics_1 = []; genetics_2 = []
        alpha = 0.1 #np.random.uniform(0.0,0.5)
        # Generate two children within the modified range of parents (Exploration + Exploitation)

        for j in range(dad.number_of_genes):
            c_max = max([dad.genes[j], mom.genes[j]])
            c_min = min([dad.genes[j], mom.genes[j]])
            h_max = c_max + alpha*(c_max - c_min)
            h_min = c_min - alpha*(c_max - c_min)
            new_gene_1, new_gene_2 = np.random.uniform(h_min,h_max,2)

            # Feasibility Assurance
            if not (self.bounds is None):
                if new_gene_1 < self.bounds[0][j]: new_gene_1 = self.bounds[0][j]
                if new_gene_1 > self.bounds[1][j]: new_gene_1 = self.bounds[1][j]
                if new_gene_2 < self.bounds[0][j]: new_gene_2 = self.bounds[0][j]
                if new_gene_2 > self.bounds[1][j]: new_gene_2 = self.bounds[1][j]

            genetics_1.append(new_gene_1)
            genetics_2.append(new_gene_2)
                
        self.new_population.append(Chromosome(genetics_1))
        self.new_population.append(Chromosome(genetics_2))

    def __Linear_CrossOver__(self, dad, mom):
        '''1. Left hand side parent
        2. Right hand side parent'''
       
        genetics_1 = []; genetics_2 = []; genetics_3 = []
        # Generate three children & choose the best 2
        for j in range(dad.number_of_genes):
            new_gene_1, new_gene_2, new_gene_3 = 0.5*(dad.genes[j] + mom.genes[j]),\
            3/2*dad.genes[j] - 0.5*mom.genes[j], 3/2*mom.genes[j] - 0.5*dad.genes[j]

            # Feasibility Assurance
            if not (self.bounds is None):
                if new_gene_1 > self.bounds[1][j]: new_gene_1 = self.bounds[1][j]
                if new_gene_2 < self.bounds[0][j]: new_gene_1 = self.bounds[0][j]
                if new_gene_2 > self.bounds[1][j]: new_gene_1 = self.bounds[1][j]
                if new_gene_3 < self.bounds[0][j]: new_gene_1 = self.bounds[0][j]
                if new_gene_3 > self.bounds[1][j]: new_gene_1 = self.bounds[1][j]

            genetics_1.append(new_gene_1)
            genetics_2.append(new_gene_2)
            genetics_3.append(new_gene_3)

        genetics_list = [genetics_1, genetics_2, genetics_3]
        fitness_list = [Chromosome(genetics_1).fitness_eval(self.obj_func), Chromosome(genetics_2).fitness_eval(self.obj_func),\
                                    Chromosome(genetics_3).fitness_eval(self.obj_func)]
        genetics_list.pop(fitness_list.index(min(fitness_list))) # Drop the least fit Genetic pattern 

        self.new_population.append(Chromosome(genetics_list[0]))
        self.new_population.append(Chromosome(genetics_list[1]))

    def __Discrete_CrossOver__(self,dad,mom):
        '''1. Left hand side parent
        2. Right hand side parent'''

        genetics_1 = []; genetics_2 = []
        # Generate two children of genes chosen discretely from the parent's pole
        for j in range(dad.number_of_genes):
            new_gene_1, new_gene_2 = np.random.choice([dad.genes[j], mom.genes[j]], size = 2)
            genetics_1.append(new_gene_1)
            genetics_2.append(new_gene_2)
                
        self.new_population.append(Chromosome(genetics_1))
        self.new_population.append(Chromosome(genetics_2))


    def __Random_Mutation__(self, i,j):
        '''1. Chromosome index
        2. Gene index'''

        self.new_population[i].genes[j] = np.random.uniform(self.bounds[0][j],self.bounds[1][j])
    
    def __Non_Uniform_Mutation__(self, i, j, b = 2):
        '''1. Chromosome index
        2. Gene index
        3. Effect of Generation Number'''
       
        taw = np.random.choice([0,1], size = 1)
        c = self.new_population[i].genes[j]

        if taw == 0:
            y = self.bounds[1][j] - c
        else:
            y = -1*(c - self.bounds[0][j])

        r = np.random.uniform(0.0,1.0)
        del_fact = y * (1 - pow(r,1-pow(self.generation/self.max_generations,b)))
        c_dash = c + del_fact

        # Feasibility Assurance
        if c_dash > self.bounds[1][j]: c_dash = self.bounds[1][j]
        if c_dash < self.bounds[0][j]: c_dash = self.bounds[0][j]
        self.new_population[i].genes[j] = c_dash

    def Initialize_GA(self, pop_size, n_o_g, max_gen = 500):
        '''Creates an initial population'''
        
        self.population_size = pop_size
        self.current_population = [Chromosome([np.random.uniform(self.bounds[0][i],self.bounds[1][i]) for i in range(n_o_g)])\
                                   for j in range(self.population_size)]
        self.generation = 1
        self.max_generations = max_gen
        
    def Evaluate_Population_Fitness(self):
        '''Evaluates the population's Fitness'''

        self.current_population_fitness = [chromosome.fitness_eval(self.obj_func) for chromosome in self.current_population]
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

        unzipped_pairs_list = list(zip(*self.parent_pairs))
        dads = np.take(self.current_population, unzipped_pairs_list[0][:]).tolist()
        moms = np.take(self.current_population, unzipped_pairs_list[1][:]).tolist()

        for i in range(len(dads)):
            prop = np.random.uniform(0.0,1.0)
            if prop > cross_prop:
                self.new_population.extend([dads[i], moms[i]])
                
            else:
                if method == 'flat':
                    self.__Flat_CrossOver__(dads[i], moms[i])
                    
                elif method == 'simple':
                    self.__Simple_CrossOver__(dads[i],moms[i])

                elif method == 'arithmetical':
                    self.__Arethmetical_CrossOver__(dads[i],moms[i])
                
                elif method == 'BLX-a':
                    self.__BLX_Alpha_CrossOver__(dads[i],moms[i])

                elif method == 'linear':
                    self.__Linear_CrossOver__(dads[i],moms[i])
                
                elif method == 'discrete':
                    self.__Discrete_CrossOver__(dads[i],moms[i])

                else:
                    self.__Simple_CrossOver__(dads[i],moms[i])

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

        if mutat_prop == 0:
            return
        
        n_o_g_Mutate = mutat_prop*self.population_size*self.global_optimum.number_of_genes
        logging.info("n_o_g_Mutate: " + str(n_o_g_Mutate))
        
        doubtful_muatations, sure_mutations = math.modf(n_o_g_Mutate)
        sure_mutations = round(sure_mutations)
        logging.info("doubtful_muatations: " + str(doubtful_muatations))
        logging.info("sure_muatations: " + str(sure_mutations))

        prop = np.random.uniform(0,1)
        logging.info("prop: " + str(prop))
        if prop < doubtful_muatations:
            sure_mutations = sure_mutations + 1
        logging.info("sure_muatations: " + str(sure_mutations))

        for n in range(sure_mutations):
            i = rnd.randint(0,self.population_size - 1)
            j = rnd.randint(0,self.global_optimum.number_of_genes - 1)     
            if method == 'random':
                self.__Random_Mutation__(i,j)
            elif method == 'non-uniform':
                self.__Non_Uniform_Mutation__(i,j)
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
            [Chromosome([np.random.uniform(self.bounds[0][i],self.bounds[1][i]) \
                         for i in range(self.current_population[0].number_of_genes)])\
                                   for j in range(self.population_size)]
        
    def __finalise_new_population__(self, refreshment_flag = 0):
        
        if refreshment_flag and self.stuck_percent >= 0.2:
            logging.info("Didn't change percent: " + str(self.stuck_percent))
            logging.info("refreshments: " + str(self.number_of_refreshments))

            self.__Refresh_GA__()
            self.stuck_percent = 0
            self.number_of_refreshments = self.number_of_refreshments + 1

        else:
            if len(self.new_population) > self.population_size:
                self.new_population = self.new_population[len(self.new_population) - self.population_size:]
                # Could be optimised to remove the worst solutions: Future Implementation
            elif len(self.new_population) < self.population_size:
                extension = [Chromosome([np.random.uniform(self.bounds[0][i],self.bounds[1][i])\
                                          for i in range(self.current_population[0].number_of_genes)])\
                                            for j in range(self.population_size - len(self.new_population))]
                self.new_population.extend(extension)
        
            self.current_population = self.new_population[:]
        self.new_population.clear()   
        self.parent_pairs.clear()


    def __Run_Algorithm__(self):
        
        self.Initialize_GA(pop_size = population_size, n_o_g = num_genes, max_gen = max_gen)
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
        # Debugging Info
        global total_evals
        total_evals = 0

        self.__Run_Algorithm__()
        logging.info("Total Evals = " + str(total_evals))
    
    def print_population(self, population):
        for i in range(len(population)):
            population[i].print_chromosome()
            print('Fitness: ', population[i].fitness)
        
        print()

