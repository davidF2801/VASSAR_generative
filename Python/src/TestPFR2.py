

import random
import matplotlib.pyplot as plt
import numpy as np

    
def is_dominated(solution1, solution2):
    science1, cost1 = solution1
    science2, cost2 = solution2

    if science1 >= science2 and cost1 <= cost2:
        return True

    return False

def compute_pareto(solutions):
    pareto_front_indices = []
    pareto_front_values = []
    
    for i in range(len(solutions)):
        is_dominated_by_others = False
        for j in range(len(solutions)):
            if i != j and is_dominated(solutions[j], solutions[i]):
                is_dominated_by_others = True
                break
        if not is_dominated_by_others:
            pareto_front_indices.append(i)
            pareto_front_values.append(solutions[i])
    
    return pareto_front_indices, pareto_front_values



def pareto_ranking(values, designs, num_pareto_fronts):
    pareto_fronts = []
    pareto_designs = []
    pareto_values = []

    values = np.array(values)
    designs = np.array(designs)

    remaining_values = values.copy()
    remaining_designs = designs.copy()

    for _ in range(num_pareto_fronts):
        pareto_front_indices, pareto_front_values = compute_pareto(remaining_values)
        pareto_front = [remaining_designs[i] for i in pareto_front_indices]

        if len(pareto_front) == 0:
            break

        pareto_fronts.append(pareto_front_indices)
        pareto_designs.append(pareto_front)
        pareto_values.append(pareto_front_values)

        remaining_values = np.delete(remaining_values, pareto_front_indices, axis=0)

    return pareto_values,pareto_designs


designs = []
num_pareto_fronts = 5
num_samples = 1000
objectives = 2
solutions = []

for _ in range(num_samples):
    sample = [random.randint(1, 1000000) for _ in range(objectives)]
    solutions.append(sample)

for _ in range(num_samples):
    sample = [random.randint(1, 1000000) for _ in range(1)]
    designs.append(sample)


pareto_values, pareto_designs = pareto_ranking(solutions, designs, num_pareto_fronts)

# Plotting the Pareto fronts
for i, front_values in enumerate(pareto_values):
    x = [val[0] for val in front_values]
    y = [val[1] for val in front_values]
    # best_designs.extend(pareto_designs[i])
    # best_sciences.extend(x)
    # best_costs.extend(y)

    plt.scatter(x, y, label='Pareto Front {}'.format(i+1))

# Add labels and show the plot
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Fronts')
plt.legend()
plt.grid(True)
plt.show()
print('Finished')
