
import matplotlib.pyplot as plt




def pareto_ranking(solutions,max_fronts):
    num_solutions = len(solutions)
    dominated_count = [0] * num_solutions
    pareto_fronts = []

    for i in range(num_solutions):
        for j in range(num_solutions):
            if i != j:
                if all(solutions[i][k] <= solutions[j][k] for k in range(len(solutions[i]))):
                    dominated_count[i] += 1

    current_front = []
    for i in range(num_solutions):
        if dominated_count[i] == 0:
            current_front.append(i)

    while current_front:
        next_front = []
        for i in current_front:
            for j in range(num_solutions):
                if i != j and solutions[j] != None:
                    if all(solutions[j][k] <= solutions[i][k] for k in range(len(solutions[j]))):
                        dominated_count[j] -= 1
                        if dominated_count[j] == 0:
                            next_front.append(j)

        pareto_fronts.append(current_front)
        current_front = next_front
        if len(pareto_fronts) >= max_fronts:  # Break out of the loop when the desired number of fronts is reached
            break

    return pareto_fronts




# def pareto_front_calculator(self, solutions, show, save = False, calculate = True):

#         if calculate:
#             costs = []
#             sciences = []
#             best_designs = []

#             for i,sol in enumerate(solutions):
#                 sol = tuple(sol)

#                 if sol in self.evaluated_designs:
#                     science_normalized, cost_normalized = self.evaluated_designs[sol]
#                     science=-science_normalized*self.science_max
#                     cost = self.cost_max*cost_normalized

#                 else:
#                     print('Evaluating design '+str(i+1)+' out of '+str(len(solutions)))

#                     science_normalized, cost_normalized = self.evaluate_design(sol)
#                     if science_normalized>0:
#                         print('Hello')
                    
#                     science=-science_normalized*self.science_max
#                     cost = self.cost_max*cost_normalized
#                     self.evaluated_designs[sol] = science_normalized, cost_normalized


                

#                 dominated = False
#                 for i, design in enumerate(best_designs):
#                     p_science, p_cost = sciences[i], costs[i]

#                     if (cost>= p_cost and science<=p_science) :
#                         dominated = True
#                         break

#                 if dominated==False:
#                         for i, design in enumerate(best_designs):
#                             p_science, p_cost = sciences[i], costs[i]
#                             if cost<=p_cost and science>=p_science:
#                                 best_designs.pop(i)
#                                 costs.pop(i)
#                                 sciences.pop(i)
#                                 self.pareto_objectives.pop(i)

#                         best_designs.append(sol)
#                         costs.append(cost)
#                         sciences.append(science)
#                         self.pareto_objectives.append((-science_normalized,cost_normalized))
                    
#             if show or save:        

#                 plt.figure('Pareto Front')
#                 # plt.scatter(savings_opt, powers_opt, c='red', label='Optimal Pareto')
#                 plt.scatter(costs, sciences,c='blue', label='Achieved Pareto')
                
#                 plt.xlabel('Cost')
#                 plt.ylabel('Science benefit')
#                 plt.legend()
#                 if save:
#                     path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'
#                     costs = np.array(costs)
#                     sciences = np.array(sciences)
#                     designs = np.array(best_designs)

#                     # Convert costs and sciences to 2D arrays
#                     costs = costs.reshape(-1, 1)
#                     sciences = sciences.reshape(-1, 1)

#                     np.save(os.path.join(path, 'initial_pareto_wl.npy'), np.hstack((costs, sciences, designs)))

#                     # Save as CSV
#                     np.savetxt(os.path.join(path, 'initial_pareto_wl.csv'), np.hstack((costs, sciences, designs)), delimiter=',')

#                     plt.savefig(os.path.join(path, 'Pareto_Front_INIT' + model_name))

#                 if show:

#                     plt.show()


#             return np.array(best_designs)
#         else:
            
#             # Specify the path to the saved files

#             path = r'C:\Users\dforn\Documents\TEXASAM\PROJECTS\VASSAR_generative\Results\PartitioningProblem\Pareto_Front'

#             # Load the data from the saved files
#             data = np.load(os.path.join(path, 'initial_pareto_part.npy'))
#             costs = data[:, 0]  # Extract the costs column
#             sciences = data[:, 1]  # Extract the sciences column
#             designs = data[:, 2:]  # Extract the designs columns
#             sciences = -sciences/self.science_max
#             costs = costs/self.cost_max
#             designs_tuple = [tuple(design) for design in designs]
#             for i,design in enumerate(designs_tuple):
#                 self.evaluated_designs[design] = sciences[i],costs[i]
#                 self.pareto_objectives.append((-sciences[i],costs[i]))


#             # # Display the loaded data
#             # print("Costs:", costs)
#             # print("Sciences:", sciences)
#             # print("Designs:", designs)
#             return designs

# Example usage:
# Suppose we have three objectives and five solutions:
# Objective 1: maximize, Objective 2: minimize, Objective 3: maximize
import random

# Generate 20 samples with 2 objectives
num_samples = 1000
objectives = 2
solutions = []

for _ in range(num_samples):
    sample = [random.randint(1, 100000) for _ in range(objectives)]
    solutions.append(sample)


pareto_fronts = pareto_ranking(solutions,5)


# Plotting the Pareto fronts
for i, front in enumerate(pareto_fronts):
    x = [solutions[index][0] for index in front]
    y = [solutions[index][1] for index in front]
    plt.scatter(x, y, label='Pareto Front {}'.format(i+1))

# Add labels and show the plot
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Fronts')
plt.legend()
plt.grid(True)
plt.show()
print('Finished')
