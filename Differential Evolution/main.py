from tests import *
from Differential_Evolution import *


if __name__ == "__main__":
    rnd.seed(time.time())

    problem = 2
    if problem  == 1:
        solution_size = 3
        max_generations = 100
        # population_size = 50
        obj_func = problem_1
        bounds = [[0,0,0], [10,10,10]]
        # Mutation_Factor = 0.8
        # XOver_Factor = 0.8

    elif problem == 2:
        solution_size = 3
        max_generations = 100
        # population_size = 50
        obj_func = controlproblem
        bounds = [[1,0.1,0.1], [100,1,0.1]]
        # Mutation_Factor = 0.8
        # XOver_Factor = 0.8

    report(solution_size, max_generations, obj_func, bounds, problem)
    
                  