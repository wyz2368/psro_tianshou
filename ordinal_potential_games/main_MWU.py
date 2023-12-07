import numpy as np
from ordinal_game_generator import runner
from MWU import multiplicative_weights_update
from Optimistic_MWU import optimistic_multiplicative_weights_update
from FLBR_MWU import FLBR_multiplicative_weights_update
from solution_solvers.nash_solver.pygambit_solver import pygbt_solve_matrix_games

def main():
    dim = 2
    num_iterations = 50
    potential, games = runner(dim)
    games.append([np.array([[2, -1], [-1, 4]]), np.array([[-2, 1], [1, -4]])])
    print("The potential is \n", potential)
    for i, game in enumerate(games):
        print("###### Running the game {} ######".format(i))
        weights = multiplicative_weights_update(game,
                                                num_iterations=num_iterations,
                                                alternate=False)
                                                
        weights1 = optimistic_multiplicative_weights_update(game,
                                                num_iterations=num_iterations,
                                                alternate=False)                                                       

        weights2 = FLBR_multiplicative_weights_update(game,
                                                num_iterations=num_iterations,
                                                alternate=False)
                                                
        ne = pygbt_solve_matrix_games(game, mode="all")
        print("Game:", game)
        print("MWU weights:", weights)
        print("OMWU weights:", weights1)
        print("FLBR weights:", weights2)
        print("NE:", ne)

main()

