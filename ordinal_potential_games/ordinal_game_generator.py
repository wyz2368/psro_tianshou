import numpy as np
from constraint import *
import random
from collections import OrderedDict

# Range of payoffs in the payoff matrix.
MIN_PAYOFF = -10
MAX_PAYOFF = 10

# Range of potentials in the potential matrix.
MIN_POTENTIAL = -10
MAX_POTENTIAL = 10


def generate_potential(n, m):
    """
    Generate a potential function.
    """
    return np.random.uniform(MIN_POTENTIAL, MAX_POTENTIAL, (n, m))

def same_sign_generator(potential, col1, row1, col2, row2):
    """
    Generate constraints for the CSP solver.
    """
    def same_sign(a, b):
        return (a - b) * (potential[row1][col1] - potential[row2][col2]) > 0
    return same_sign


def generate_ordinal_potential_game(potential, num_sols=5, mode="one"):
    """
    Given a potential function, generate an ordinal potential game by solving a CSP.
    """
    n,m = np.shape(potential)

    # problem = Problem(solver=MinConflictsSolver)
    problem = Problem()
    problem.addVariables(range(n*m), range(MIN_PAYOFF, MAX_PAYOFF + 1))
    for col in range(m):
        for row in range(n):
            for subrow in range(row + 1, n):
                constraint = same_sign_generator(potential, col, row, col, subrow)
                problem.addConstraint(constraint, [row * m + col, subrow * m + col])

    for row in range(n):
        for col in range(m):
            for subcol in range(col + 1, m):
                constraint = same_sign_generator(potential, col, row, subcol, row)
                problem.addConstraint(constraint, [row * m + col, row * m + subcol])

    if mode == "all":
        solutions = problem.getSolutions()
    else:
        solutions = []
        sol_iter = problem.getSolutionIter()
        for i in range(num_sols):
            solutions.append(next(sol_iter))

    return solutions


def observer(n, m, solution):
    """
    Formulate a solution to a payoff matrix.
    """
    payoff_matrix = np.zeros((n, m))
    # print(solution)
    for key in solution:
        col = key % m
        row = key // m
        payoff_matrix[row][col] = solution[key]

    return payoff_matrix

def make_payoff_matrix(n, m, solutions):
    payoff_matrix = []
    for i in range(2):
        sampled_solution = random.sample(solutions, 1)
        # print(solutions)
        payoff_matrix.append(observer(n, m, *sampled_solution))

    return payoff_matrix

def find_maxmin_matrix(n, m, solutions):
    """
    Find a matrix that max min (x_{ij} - x_{ik}) - (phi_{ij} - phi_{ik})
    """
    value_matrix = OrderedDict()
    for solution in solutions:
        matrix = observer(n, m, solution)
        min_value = compute_min_value(matrix)
        while min_value in value_matrix:
            min_value -= 0.001
        value_matrix[min_value] = matrix

    sort_values = sorted(value_matrix, reverse=True)
    result_matrix = []
    for value in sort_values:
        result_matrix.append(value_matrix[value])

    return result_matrix

def make_minmax_payoff_matrix(solutions):
    games = []
    for num_games in range(5):
        payoff_matrix = []
        for i in range(2):
            if i == 0:
                sampled_solution = random.choice(solutions)
            else:
                sampled_solution = random.choice(solutions[:int(0.5 * len(solutions))])
            payoff_matrix.append(sampled_solution)

        games.append(payoff_matrix)

    return games


def compute_min_value(payoff_matrix):
    min_value = 999999
    n,m = np.shape(payoff_matrix)
    for i in range(n):
        sorted_row = sorted(payoff_matrix[i, :])
        value = sorted_row[-1] - sorted_row[0]
        if value < min_value:
            min_value = value
    for j in range(m):
        sorted_col = sorted(payoff_matrix[:, j])
        value = sorted_col[-1] - sorted_col[0]
        if value < min_value:
            min_value = value

    return min_value

def runner(dim):
    potential = generate_potential(dim, dim)
    n, m = np.shape(potential)
    solutions = generate_ordinal_potential_game(potential, mode="all")
    result_matrix = find_maxmin_matrix(n, m, solutions)
    games = make_minmax_payoff_matrix(result_matrix)

    return potential, games




if __name__ == "__main__":
    # Hard coding a potential:
    # potential = np.array([[1,2,3], [4,5,6], [7,8,9]])
    # Randomly generate a potential
    dim = 6
    potential = generate_potential(dim, dim)
    n, m = np.shape(potential)
    solutions = generate_ordinal_potential_game(potential, num_sols=5)
    # print(observer(n, m, *solutions))
    print("The potential is \n", potential)
    print("solutions:", solutions)
    # result_matrix = find_maxmin_matrix(n, m, solutions)
    # payoff_matrix = make_minmax_payoff_matrix(result_matrix)
    # # print(make_payoff_matrix(n, m, solutions))
    # print("The generated payoff matrix is \n", payoff_matrix)
