import numpy as np
# import random
# import functools
# from scipy.stats import entropy
import logging


def init_logger(logger_name, checkpoint_dir):
    # Set up logging info.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(checkpoint_dir + "/data.log")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def general_get_joint_strategy_from_marginals(probabilities):
    """Returns a joint strategy matrix from a list of marginals.
    Does not require marginals to have the same lengths.
    Args:
      probabilities: list of probabilities.

    Returns:
      A joint strategy from a list of marginals
    """
    joint = np.outer(probabilities[0], probabilities[1])
    for i in range(len(probabilities) - 2):
        joint = joint.reshape(tuple(list(joint.shape) + [1])) * probabilities[i + 2]
    return joint

def mixed_strategy_payoff(meta_games, probs):
    """
    A multiple player version of mixed strategy payoff writen below by yongzhao
    The lenth of probs could be smaller than that of meta_games
    """
    assert len(meta_games) == len(probs),'number of player not equal'
    for i in range(len(meta_games)):
        assert len(probs[i]) <= meta_games[0].shape[i],'meta game should have larger dimension than marginal probability vector'
    prob_matrix = general_get_joint_strategy_from_marginals(probs)
    prob_slice = tuple([slice(prob_matrix.shape[i]) for i in range(len(meta_games))])
    meta_game_copy = [ele[prob_slice] for ele in meta_games]
    payoffs = []

    for i in range(len(meta_games)):
        payoffs.append(np.sum(meta_game_copy[i] * prob_matrix))
    return payoffs

# This older version of function must be of two players
def mixed_strategy_payoff_2p(meta_games, probs):
   payoffs = []
   prob1 = probs[0]
   prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
   prob2 = probs[1]
   for meta_game in meta_games:
       payoffs.append(np.sum(prob1 * meta_game * prob2))
   return payoffs

def find_all_deviation_payoffs(empirical_games, meta_game, caches):
    """
    Find all deviation payoff of pure strategy profile. Only need to calculate
    sum_i|S_i| deviations. Only for 2-player game.
    :param empirical_games:
    :param meta_game: the underlying true game
    :param caches: storage of deviation payoffs.
    :param mean:
    :return:
    """
    num_strategies_p0 = len(empirical_games[0])
    num_strategies_p1 = len(empirical_games[1])

    if num_strategies_p0 != num_strategies_p1:
        raise ValueError("Haven't supported that 2 players have different number of strategies.")

    # Allow redundant strategies.
    diagonal_profiles = list(zip(empirical_games[0], empirical_games[1]))
    for profile in diagonal_profiles:
        _, payoff = deviation_pure_strategy_profile(meta_game, profile)
        caches[0].save(key=profile[1], value=payoff[0])
        caches[1].save(key=profile[0], value=payoff[1])

    return caches


def deviation_pure_strategy_profile(meta_games, strategis):
    """
    Find the deviation strategy and payoff for pure strategy profile.
    For 2-player case only.
    :param meta_games: the full game matrix.
    :param strategis: [strategy idx for p1, strategy idx for p2]
    :return:
    """
    dev_strs = []
    dev_strs.append(np.argmax(meta_games[0][:, strategis[1]]))
    dev_strs.append(np.argmax(meta_games[1][strategis[0], :]))

    dev_payoff = [meta_games[0][dev_strs[0], strategis[1]], meta_games[1][strategis[0], dev_strs[1]]]

    return dev_strs, dev_payoff

def benefitial_deviation_pure_strategy_profile(meta_games, opponent, strategy, base_value):
    """
    Find the deviation strategy and payoff for pure strategy profile.
    For 2-player case only.
    :param meta_games: the full game matrix.
    :param strategis: [strategy idx for p1, strategy idx for p2]
    :return:
    """
    if opponent == 1:
        payoff_vec = meta_games[0][:, strategy]
        payoff_vec = np.reshape(payoff_vec, -1)
        idx = list(np.where(payoff_vec > base_value[0])[0])

    else:
        payoff_vec = meta_games[1][strategy, :]
        payoff_vec = np.reshape(payoff_vec, -1)
        idx = list(np.where(payoff_vec > base_value[1])[0])

    return payoff_vec[idx]


def deviation_strategy(meta_games, probs):
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    return dev_strs, dev_payoff

# Functions for 3-player games.
def dev_regret_general(meta_games, probs):
    """
        Calculate the regret of a profile in an empirical game with any number of players.
        :param meta_games:
        :param probs: a strategy profile
        :return:
        """
    num_players = len(meta_games)
    num_strategies = np.shape(meta_games[0])

    prob_matrix = general_get_joint_strategy_from_marginals(probs)
    deviation_payoffs = []

    for i in range(num_players):
        profile_payoff = np.sum(meta_games[i] * prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(num_strategies[i]):
            dev_prob = probs.copy()
            dev_prob[i] = np.zeros(num_strategies[i])
            dev_prob[i][j] = 1
            new_prob_matrix = general_get_joint_strategy_from_marginals(dev_prob)
            dev_payoff.append(np.sum(meta_games[i] * new_prob_matrix))
        deviation_payoffs.append(dev_payoff - profile_payoff)

    dev_strs = [np.argmax(ele) for ele in deviation_payoffs]
    dev_payoff = []
    for idx, str in enumerate(dev_strs):
        dev_payoff.append(deviation_payoffs[idx][str])

    nashconv = np.sum([np.max(ele) for ele in deviation_payoffs])

    return dev_strs, dev_payoff, nashconv

    # nashconv = np.sum([np.max(ele) for ele in deviation_payoffs])
    # return nashconv


def project_onto_unit_simplex(prob):
    """
    Project an n-dim vector prob to the simplex Dn s.t.
    Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
    :param prob: a numpy array. Each element is a probability.
    :return: projected probability
    """
    prob_length = len(prob)
    bget = False
    sorted_prob = -np.sort(-prob)
    tmpsum = 0

    for i in range(1, prob_length):
        tmpsum = tmpsum + sorted_prob[i-1]
        tmax = (tmpsum - 1) / i
        if tmax >= sorted_prob[i]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + sorted_prob[prob_length-1] - 1) / prob_length

    return np.maximum(0, prob - tmax)

def beneficial_deviation(meta_games, probs, base_value):
    """
    Find all beneficial deviations and corresponding payoffs. (For 2-player game only.)
    :param meta_games:
    :param probs:
    :param base_value: deviation beyond this value. [p1, p2]
    :return:
    """
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = list(np.where(payoff_vec > base_value[0])[0])
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = list(np.where(payoff_vec > base_value[1])[0])
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    return dev_strs, dev_payoff

def sample_deviation_strategy(dev_strs, dev_payoff):
    """
    Sample a deviation strategy and corresponding payoff.
    :param dev_strs:
    :param dev_payoff:
    :return:
    """
    num_players = len(dev_strs)
    num_deviations = len(dev_strs[0])
    sampled_str = []
    sample_payoff = []

    for player in range(num_players):
        idx = np.random.choice(np.arange(num_deviations))
        sampled_str.append(dev_strs[player][idx])
        sample_payoff.append(dev_payoff[player][idx])

    return sampled_str, sample_payoff

def deviation_within_EG(meta_games, empirical_games, probs):
    """
    Calculate the deviation payoff of a profile within a empirical game. (For 2-player game only.)
    :param meta_games:
    :param empirical_games:
    :param probs:
    :return:
    """

    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    payoff_vec_EG = payoff_vec[empirical_games[0]]
    dev_payoff.append(np.max(payoff_vec_EG))

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    payoff_vec_EG = payoff_vec[empirical_games[1]]
    dev_payoff.append(np.max(payoff_vec_EG))

    return dev_payoff
