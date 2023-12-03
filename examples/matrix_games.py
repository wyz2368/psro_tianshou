import numpy as np

from matrix_game_analysis.psro4matrix import psro4matrix
from matrix_game_analysis.utils import init_logger
import os
from absl import app
from absl import flags
import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string("meta_games_path", './meta_games.npy', "Path to npy meta_games")
flags.DEFINE_string("root_result_folder",'root_result',"root directory of saved results")
flags.DEFINE_string("meta_strategy_method", "DO",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("psro_iterations", 3,
                     "Number of training steps for PSRO.")
def psro_runner(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Set up file path.
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)
    checkpoint_dir = "matrix_example" + '_'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # meta_games = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
    meta_games = np.load(FLAGS.meta_games_path)

    logger = init_logger(__name__, checkpoint_dir)
    logger.info("============ Begin running PSRO for Matrix Analysis ==========")
    logger.info("The meta-strategy solver is {}.".format(FLAGS.meta_strategy_method))
    logger.info("The number of iteration is {}.".format(FLAGS.psro_iterations))
    logger.info("The game dimension is {}.".format(np.shape(meta_games)[1:]))


    psro4matrix(meta_games=meta_games,
                meta_strategy_string=FLAGS.meta_strategy_method,
                checkpoint_dir=checkpoint_dir,
                num_iterations=FLAGS.psro_iterations)

    logger.info("============ End running PSRO for Matrix Analysis ==========")


if __name__ == "__main__":
    app.run(psro_runner)