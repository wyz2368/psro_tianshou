import collections
from solution_solvers.nash_solver.projected_replicator_dynamics import projected_replicator_dynamics
from solution_solvers.nash_solver.pygambit_solver import pygbt_solve_matrix_games
from solution_solvers.nash_solver.lp_solver import lp_solve
from matrix_game_analysis.utils import *

import numpy as np

def double_oracle(meta_games, empirical_games, checkpoint_dir, gambit=True, sample_dev=False):
    """
    Double oracle method.
    :param meta_games:
    :param empirical_games:
    :param checkpoint_dir:
    :param gambit: Whether using gambit as a Nash solver. False： lp solver
    :param sample_dev: Whether sample a beneficial deviation or sample an argmax deviation
    :return:
    """
    num_players = len(meta_games)
    num_strategies, _ = np.shape(meta_games[0])
    subgames = []

    idx0 = sorted(list(set(empirical_games[0])))
    idx1 = sorted(list(set(empirical_games[1])))
    idx = np.ix_(idx0, idx1)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    if gambit:
        # Gambit solver
        # nash = gambit_solve(subgames, mode="one", checkpoint_dir=checkpoint_dir)
        nash = pygbt_solve_matrix_games(subgames, method="lcp", mode="one")[0] #TODO: Test the [0]
    else:
        # LP solver
        nash = lp_solve(subgames)

    nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate([idx0, idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    if sample_dev:
        dev_strs, dev_payoff = beneficial_deviation(meta_games, meta_game_nash, nash_payoffs)
        dev_strs, dev_payoff = sample_deviation_strategy(dev_strs, dev_payoff)
    else:
        dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)

    nashconv = 0
    for player in range(num_players):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv, nash

def fictitious_play(meta_games, empirical_games, checkpoint_dir=None, sample_dev=False):
    num_strategies, _ = np.shape(meta_games[0])
    subgames = []
    counter0 = collections.Counter(empirical_games[0])
    counter1 = collections.Counter(empirical_games[1])

    idx0 = sorted(list(set(empirical_games[0])))
    idx1 = sorted(list(set(empirical_games[1])))
    idx = np.ix_(idx0, idx1)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    nash0 = np.ones(len(idx0))
    for i, item in enumerate(idx0):
        nash0[i] = counter0[item]
    nash0 /= np.sum(nash0)

    nash1 = np.ones(len(idx1))
    for i, item in enumerate(idx1):
        nash1[i] = counter1[item]
    nash1 /= np.sum(nash1)
    nash = [nash0, nash1]
    
    nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate([idx0, idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    # dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)
    if sample_dev:
        dev_strs, dev_payoff = beneficial_deviation(meta_games, meta_game_nash, nash_payoffs)
        dev_strs, dev_payoff = sample_deviation_strategy(dev_strs, dev_payoff)
    else:
        dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)

    nashconv = 0
    for player in range(len(meta_games)):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv, nash


def prd_solver(meta_games, empirical_games, checkpoint_dir=None):
    num_players = len(meta_games)
    num_strategies = np.shape(meta_games[0])[0]
    subgames = []

    sub_idx = []
    for player in range(num_players):
        sub_idx.append(sorted(list(set(empirical_games[player]))))

    idx = np.ix_(*sub_idx)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    nash = projected_replicator_dynamics(subgames)

    # nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate(sub_idx):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    dev_strs, dev_payoff, nashconv = dev_regret_general(meta_games, meta_game_nash)

    return dev_strs, nashconv, nash















