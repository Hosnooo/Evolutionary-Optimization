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

class Differential_Evolution():

    def __init__(self, solution_size, max_generations, population_size, obj_func = None, bounds = None,\
                 Mutation_Factor = 0.8, XOver_Factor = 0.8):

        # Sizes
        self.solution_size = solution_size
        self.population_size = population_size
        self.max_generations = max_generations

        # Solution packets
        self.current_population = []
        self.current_population_cost = []
        self.mutants = []
        self.mutants_cost = []
        self.trials = []
        self.trials_cost = []
        self.best_solution = []
        self.minimum_cost = np.inf
        self.minimum_cost_history = []
        self.current_best_solution_idx = []
        self.current_best_cost_history = []

        # Objective Function
        self.obj_func = obj_func
        self.bounds = bounds
        
        # Algorithm Parameters
        self.Mutation_Factor = Mutation_Factor
        self.XOver_Factor = XOver_Factor
        
        # Termination Parameters
        self.current_generation = 0
        self.stuck_percent = 0
  
    def cost_eval(self, solution):
        
        cost = self.obj_func(solution)
        if cost == 0:
            cost = 1e-20     # Case: division by Zero
        
        return cost
    
    def Initialize_DE(self):
        '''Creates an initial population'''

        for j in range(self.population_size):
            solution = [np.random.uniform(self.bounds[0][i],self.bounds[1][i]) for i in range(self.solution_size)]
            self.current_population.append(solution)
            self.current_population_cost.append(self.cost_eval(solution))
                    
    
    def Update_Best_Solution(self):
        '''Updates Global Optimum'''

        self.current_best_solution_idx = self.current_population_cost.index(min(self.current_population_cost))
        self.current_best_cost_history.append(self.current_population_cost[self.current_best_solution_idx])

        if self.current_population_cost[self.current_best_solution_idx] < self.minimum_cost:
            self.best_solution = self.current_population[self.current_best_solution_idx].copy()
            self.minimum_cost = self.current_population_cost[self.current_best_solution_idx]
            self.stuck_percent = 0

        else:
            self.stuck_percent = ((self.stuck_percent*self.max_generations) + 1)/self.max_generations
                    
        self.minimum_cost_history.append(self.minimum_cost)

    def Mutate(self):
        '''Generates mutant population (embryos of the new population)'''

        self.mutants.clear()
        self.mutants_cost.clear()

        for i in range(self.population_size):
            
            mutant = np.array(self.current_population[rnd.randint(0,self.population_size-1)]) + self.Mutation_Factor *\
                    (np.array(self.current_population[rnd.randint(0,self.population_size-1)]) - \
                    np.array(self.current_population[rnd.randint(0,self.population_size-1)]))
            
            self.mutants.append(mutant.tolist())
            self.mutants_cost.append(self.cost_eval(self.mutants[i]))

        self.current_generation = self.current_generation + 1
        
    
    def XOver(self):
        '''Crosses Over the mutants with The current population'''

        self.trials.clear()
        self.trials_cost.clear()

        for i in range(self.population_size):
            prop = np.random.uniform(0.0,1.0)
            if prop > self.XOver_Factor:
                self.trials.append(self.current_population[i])
                self.trials_cost.append(self.current_population_cost[i])
            else:
                self.trials.append(self.mutants[i])
                self.trials_cost.append(self.mutants_cost[i])

    def Select(self):
        '''Selects between the current population solutions and the mutated-xovered solutions'''

        for i in range(self.population_size):
            if not (self.trials_cost[i] > self.current_population_cost[i]):
                self.current_population[i] = self.trials[i]
                self.current_population_cost[i] = self.trials_cost[i]

    def Run(self):
        '''Main algorithm fLow drive'''

        self.Initialize_DE()

        while 1:

            self.Update_Best_Solution()
            if self.stuck_percent > 0.2 or self.current_generation == self.max_generations:
                return
            
            self.Mutate()
            self.XOver()
            self.Select()