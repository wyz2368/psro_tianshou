from ordinal_game_generator import runner
from MWU import multiplicative_weights_update
from solution_solvers.nash_solver.pygambit_solver import pygbt_solve_matrix_games

def main():
    dim = 2
    num_iterations = 10
    potential, games = runner(dim)
    print("The potential is \n", potential)
    for i, game in enumerate(games):
        print("###### Running the game {} ######".format(i))
        weights = multiplicative_weights_update(game,
                                                num_iterations=num_iterations,
                                                alternate=False)

        ne = pygbt_solve_matrix_games(game, mode="all")
        print("Game:", game)
        print("MWU weights:", weights)
        print("NE:", ne)

if __name__ == "__main__":
    main()

