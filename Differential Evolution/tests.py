from Differential_Evolution import *

abs_path = os.path.abspath(__file__); results_dir = os.path.join(os.path.dirname(abs_path), 'Results/') # Path for saving Results.

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

def problem_1(vars):
    cost = pow(vars[0],2) + 2*pow(vars[1],2) + 3*pow(vars[2],2) \
        + vars[0]*vars[1] + vars[1]*vars[2] - 8*vars[0] - 16*vars[1] - 32*vars[2] + 110
    
    return cost

def report(solution_size, max_generations, func, bounds, case):

    results_dir = os.path.join(os.path.dirname(abs_path), 'Results/' + str(case) + '/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    repititions = 10

    best_history = []; sols = []; iterations_list = []; costs = []; params = []; gens = []; best_costs = [];
    for i in range(repititions):

        Mutation_Factor, XOver_Factor, population_size = np.random.uniform(0.4,1),\
              np.random.uniform(0.5,0.9), rnd.randint(30,60)
        
        params.append([Mutation_Factor, XOver_Factor, population_size])

        problem = Differential_Evolution(solution_size, max_generations, population_size, func, bounds,\
                 Mutation_Factor, XOver_Factor)
        problem.Run()

        sols.append(problem.best_solution)
        costs.append(problem.minimum_cost_history)
        best_costs.append(problem.minimum_cost)

        iterations = list(range(0, problem.current_generation+1))
        iterations_list.append(iterations)
        gens.append(problem.current_generation)


    plt.figure(1)
    for i in range(len(iterations_list)):
        plt.plot(iterations_list[i], costs[i], '-.')

    if func == controlproblem:    
        plt.xlabel('Generation'); plt.ylabel('Maximum Sigma')
    elif func == problem_1:
        plt.xlabel('Generation'); plt.ylabel('F(x)')

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
    table.append(best_costs)
    table.append(gens)

    df = pd.DataFrame(table).T
    headers = ["Mutation Factor", "XOver Factor", "Population Size", "1", "2", "3", "Cost", "Generations"]
    df.columns = headers

    if func == controlproblem:
        poles_list = np.array([compute_poles(sols[i]) for i in range(len(sols[:,0]))])
        poles_headers = ['pole%s' %i for i in range(len(poles_list[0,:]))]

        for i in  range(len(poles_list[0,:])):
            df['pole%s' %i] = poles_list[:,i]

        headers.extend(poles_headers)
        
    print(df)
    df.to_csv(results_dir + 'output_control%s.csv' %str(case), index=True, header=headers)