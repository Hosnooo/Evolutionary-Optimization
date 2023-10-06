from tests import *
from RC_Genetic_Algorithm import *

__FILE_BASE__ = r"E:\KFUPM - Masters\1\Intelligence Control\Genetic Algorithm\Results\log.txt"
if os.path.exists(__FILE_BASE__):
  os.remove(__FILE_BASE__)

logging.basicConfig(filename=__FILE_BASE__, level=logging.INFO)
logging.debug("Debug logging test...")
log_flag = 1 # log file flag




if __name__ == "__main__":
    rnd.seed(time.time())

    if 0: 
        if 0:
            report_generator(param='x-over')
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
    elif 0:
        if not refreshment_flag:
            logging.info("No Refreshment")
        basic_runs()

    else:
        control_problem_report(bounds=[[1, 0.1, 0.1], [30, 0.5, 0.1]], case= 'case2')
