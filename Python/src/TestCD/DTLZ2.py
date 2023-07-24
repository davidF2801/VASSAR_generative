import numpy as np

from pymop.factory import get_problem


def dtlz2(x):

    problem = get_problem("dtlz2", n_var=len(x), n_obj=2)

    f = tuple(problem.evaluate(x)/1.5)

    return f
    # def dtlz2(x, num_objectives):
#     # Number of decision variables
#     num_variables = len(x)

#     g = 0
#     for i in range(num_variables - num_objectives + 1, num_variables):
#         g += (x[i] - 0.5) ** 2 - np.cos(20 * np.pi * (x[i] - 0.5))

#     g = 100 * (num_variables - num_objectives + 1 + g)

#     objectives = []
#     for i in range(num_objectives):
#         obj = (1 + g) * 0.5
#         for j in range(num_objectives - i - 1):
#             obj *= x[j]
#         if i != 0:
#             obj *= 1 - x[num_objectives - i - 1]
#         objectives.append(obj)

#     return objectives

