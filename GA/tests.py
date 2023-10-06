
from RC_Genetic_Algorithm import *
abs_path = os.path.abspath(__file__); results_dir = os.path.join(os.path.dirname(abs_path), 'Results/') # Path for saving Results.


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
    repetitions = 0
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
    table = [params[:,i] for i in range(len(params[0,:]))]; table.extend([sols[:,i] for i in range(len(sols[0,:]))])
    table.append(fitn)
    df = pd.DataFrame(table).T
    print(df)
    df.to_csv(results_dir + 'output_basic.csv', index=False, header=False)


def controlproblem(gains):
    # Define matrices A and B
    A = np.array([[0, 377, 0, 0],
                [-0.0587, 0, -0.1303, 0],
                [-0.0899, 0, -0.1956, 0.1289],
                [95.605, 0, -816.0862, -20]])

    B = np.array([[0, 0, 0, 0],
                [0, 0, 0, 1000]]).T

    # Calculate KCL and BCL matrices
    KCL = np.array([[-0.0587, 0, -0.1303, 0],
                    [-0.0587 * gains[0] * gains[1] / gains[2], 0, -0.1303 * gains[0] * gains[1] / gains[2], 0]])

    BCL = np.array([[-0.333, 0],
                    [gains[0] / gains[2] * (1 - gains[1] / 3), -1 / gains[2]]])

    # Concatenate matrices to form Ac
    Ac = np.block([[A, B],
                [KCL, BCL]])

    # Calculate closed-loop eigenvalues
    closedloopeigenvalues = np.linalg.eigvals(Ac)
    if np.any(np.real(closedloopeigenvalues) > 0):
        return -1e-20
    
    # Find real indices of eigenvalues
    cmplx_idxs = np.where(np.imag(closedloopeigenvalues) != 0)

    # Extract real eigenvalues
    sigmas = np.real(closedloopeigenvalues)
    osci_modes_sigmas = sigmas[cmplx_idxs]

    # Calculate the maximum real part (cost)
    return max(osci_modes_sigmas)

def compute_poles(gains):
    # Define matrices A and B
    A = np.array([[0, 377, 0, 0],
                [-0.0587, 0, -0.1303, 0],
                [-0.0899, 0, -0.1956, 0.1289],
                [95.605, 0, -816.0862, -20]])

    B = np.array([[0, 0, 0, 0],
                [0, 0, 0, 1000]]).T

    # Calculate KCL and BCL matrices
    KCL = np.array([[-0.0587, 0, -0.1303, 0],
                    [-0.0587 * gains[0] * gains[1] / gains[2], 0, -0.1303 * gains[0] * gains[1] / gains[2], 0]])

    BCL = np.array([[-0.333, 0],
                    [gains[0] / gains[2] * (1 - gains[1] / 3), -1 / gains[2]]])

    # Concatenate matrices to form Ac
    Ac = np.block([[A, B],
                [KCL, BCL]])

    # Calculate closed-loop eigenvalues
    return np.linalg.eigvals(Ac)

def control_problem_report(bounds = [[1, 0.1, 0.01], [30, 1, 0.1]], case = 'case1'):
    global strong_to_win, crossover_prop, mutation_prop, results_dir, population_size
    
    results_dir = os.path.join(os.path.dirname(abs_path), 'Results/control/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    repititions = 10

    global_histories = []; global_sols = []; iterations_list = []; fitn = []; sols = []; params = []; gens = []
    for i in range(repititions):
        #first_run_params = [0.9, 0.8, 0.01]
        strong_to_win, crossover_prop, mutation_prop, population_size = np.random.uniform(0.7,0.9),\
              np.random.uniform(0.8,1), np.random.uniform(0.01,0.05), rnd.randint(30,60)
        
        params.append([strong_to_win, crossover_prop, mutation_prop, population_size])

        problem = Genetic_Algorithm(obj_func = controlproblem, bounds = bounds)
        problem.Run()
        sols.append(problem.global_optimum.genes)
        fitn.append(1/problem.global_optimum.fitness)
        global_histories.append(np.reciprocal(problem.global_optimum_history))
        global_sols.append(problem.global_optimum.genes)
        iterations = list(range(1, problem.generation))
        iterations_list.append(iterations)
        gens.append(problem.generation)

    printer = "strong_to_win, crossover_prop, mutation_prop, pop_size = %.3f, %.3f, %.4f, %i"\
          %(strong_to_win,crossover_prop,mutation_prop,population_size)
    plt.figure(1)
    #plt.plot(list(range(1,problem.generation)), problem.best_individual_history, label='Local Optimum Fitness')
    for i in range(len(iterations_list)):
        plt.plot(iterations_list[i], global_histories[i], '-.')
    plt.xlabel('Generation'); plt.ylabel('Maximum Sigma')
    plt.legend()
    plt.grid()
    plt.title("Solution Convergence VS Generation")
    plt.legend(list(range(1,repititions+1)))
    #plt.show() 
    plt.savefig(results_dir + "Global_conv%s.png" %str(case))
    plt.clf()
    
    params = np.array(params)
    sols = np.array(sols)

    table = [params[:,i] for i in range(len(params[0,:]))]; table.extend([sols[:,i] for i in range(len(sols[0,:]))])
    table.append(fitn)
    table.append(gens)
    poles_list = np.array([compute_poles(sols[i]) for i in range(len(sols[:,0]))])
    #table.extend([poles_list[:,i] for i in range(len(poles_list[0,:]))])
    poles_headers = ['pole%s' %i for i in range(len(poles_list[0,:]))]

    headers = ["strong_to_win", "crossover_prop", "mutation_prop", "pop_size", "K", "T1", "T2", "Fitness", "Generations"]
    df = pd.DataFrame(table).T
    df.columns = headers

    for i in  range(len(poles_list[0,:])):
        df['pole%s' %i] = poles_list[:,i]

    print(df)
    headers.extend(poles_headers)
    df.to_csv(results_dir + 'output_control%s.csv' %str(case), index=True, header=headers)