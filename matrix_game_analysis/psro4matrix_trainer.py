import numpy as np
from matrix_game_analysis.utils import init_logger
# import logging
# logger = logging.getLogger(__name__)
# import functools
# print = functools.partial(print, flush=True)

class PSRO4matrix_trainer(object):
    def __init__(self,
                 meta_games,
                 meta_method,
                 checkpoint_dir,
                 num_iterations,
                 init_strategies=None):
        """
        Inputs:
            num_rounds      : repeat psro_lib on matrix games from #num_rounds start points
            meta_method_list: for heuristics block switching
            seed            : an integer.
            init_strategies : a len(num_rounds) list or a number
        """
        self.meta_games = meta_games
        self.meta_method = meta_method
        self.num_strategies = [len(meta_games[i]) for i in range(len(meta_games))]
        self.checkpoint_dir = checkpoint_dir
        self.num_players = len(meta_games)
        if init_strategies is not None:
            self.init_strategies = init_strategies
        else:
            self.init_strategies = [np.random.randint(0, len(meta_games[i])) for i in range(len(meta_games))]

        self.empirical_games = [[] for _ in range(self.num_players)]
        self.num_iterations = num_iterations

        # Logging info
        self.logger = init_logger(logger_name=__name__,
                                  checkpoint_dir=checkpoint_dir)


        self.logger.info("============ Begin running PSRO trainer ===========")
        self.logger.info("Initial strategies: {}".format(self.init_strategies))


    def init_round(self):
        self.empirical_games = [[self.init_strategies[i]] for i in range(self.num_players)]

    def iteration(self):
        for it in range(self.num_iterations):
            dev_strs, nashconv, meta_probs = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            self.logger.info('################## Iteration {} ###################'.format(it + 1))
            self.logger.info("Current empirical game: {}".format(self.empirical_games))
            sorted_empirical_games = [sorted(list(set(self.empirical_games[i]))) for i in range(len(self.meta_games))]
            self.logger.info("Sorted empirical game: {}".format(sorted_empirical_games))
            self.logger.info("Meta-strategy probability: {}".format(meta_probs))
            self.logger.info("The current regret: {}".format(nashconv))

            if nashconv < 1e-6:
                break

            for i in np.arange(self.num_players):
                self.empirical_games[i].append(dev_strs[i])
                self.empirical_games[i] = sorted(self.empirical_games[i])

    def loop(self):
        self.init_round()
        self.iteration()

    def get_empirical_game(self):
        return self.empirical_games

