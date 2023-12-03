from psro_lib import meta_strategies
from solution_solvers.nash_solver.pygambit_solver import pygbt_solve_matrix_games
from psro_lib import utils

import numpy as np
import itertools
import pickle
import os


def regret_of_last_iter(meta_games, meta_probs, expected_payoffs):
    regrets = []
    meta_game0 = np.copy(meta_games[0])
    prob = np.append(meta_probs[1], [0.0])
    regrets.append(np.max(meta_game0.dot(prob)) - expected_payoffs[0])

    prob = np.append(meta_probs[0], [0.0])
    meta_game1 = np.copy(meta_games[1]).transpose()
    regrets.append(np.max(meta_game1.dot(prob)) - expected_payoffs[1])

    return regrets



def regret(meta_games, subgame_index, subgame_ne=None, start_index=0):
    """
    (Only used in block switch.)
    Calculate the regret based on a complete payoff matrix for PSRO
    In subgame, each player could have different number of strategies
    :param meta_games: meta_games in PSRO
    :param subgame_index: last policy index in subgame.
                          subgame_index-start_index+1=number of policy
                          int/list. If int, players have same num of strategies
    :param start_index: starting index for the subgame.
                          int/list. If int, assume subgame in all num_players dimension
                          have the same index
    :param: subgame_ne: subgame nash equilibrium vector.
    :return: a list of regret, one for each player.
    """
    num_policy = np.array(np.shape(meta_games[0]))
    num_players = len(meta_games)
    subgame_index = np.ones(num_players, dtype=int) * subgame_index \
        if np.isscalar(subgame_index) else subgame_index
    start_index = np.ones(num_players, dtype=int) * start_index \
        if np.isscalar(start_index) else start_index
    if not sum(num_policy != subgame_index - start_index + 1):
        print("The subgame is same as the full game. Return zero regret.")
        return np.zeros(num_players)

    num_new_pol_back = num_policy - subgame_index - 1
    index = [list(np.arange(start_index[i], subgame_index[i] + 1)) for i in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]

    # Change the "lcp" to "gnm" for games with more than 2 players.
    nash = pygbt_solve_matrix_games(meta_games, method="lcp", mode="one") if not subgame_ne else subgame_ne
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(nash)
    this_meta_prob = [
        np.concatenate(([0 for _ in range(start_index[i])], nash[i], [0 for _ in range(num_new_pol_back[i])])) for i in
        range(num_players)]
    nash_payoffs = []
    deviation_payoffs = []

    for i in range(num_players):
        ne_payoff = np.sum(submeta_games[i] * prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(start_index[i] + num_new_pol_back[i]):
            dev_prob = this_meta_prob.copy()
            dev_prob[i] = np.zeros(num_policy[i])
            if j < start_index[i]:
                dev_prob[i][j] = 1
            else:
                dev_prob[i][subgame_index[i] + j - start_index[i] + 1] = 1
            new_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
            dev_payoff.append(np.sum(meta_games[i] * new_prob_matrix))
        deviation_payoffs.append(dev_payoff - ne_payoff)
        nash_payoffs.append(ne_payoff)

    regret = [np.max(ele) for ele in deviation_payoffs]
    return regret


def strategy_regret(meta_games, subgame_index, ne=None, subgame_ne=None):
    """
        Calculate the strategy regret based on a complete payoff matrix for PSRO.
        strategy_regret of player equals to nash_payoff in meta_game - fix opponent nash strategy, player deviates to subgame_nash
        Assume all players have the same number of policies.
        :param meta_games: meta_games in PSRO
        :param subgame_index: subgame to evaluate, redundant if subgame_nash supplied
        :param: nash: equilibrium vector
        :param: subgame_ne: equilibrium vector
        :return: a list of regret, one for each player.

    """
    num_players = len(meta_games)
    num_new_pol = np.shape(meta_games[0])[0] - subgame_index

    # Change the "lcp" to "gnm" for games with more than 2 players.
    ne = pygbt_solve_matrix_games(meta_games, method="lcp", mode="one") if not ne else ne
    index = [list(np.arange(subgame_index)) for _ in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]

    # Change the "lcp" to "gnm" for games with more than 2 players.
    subgame_ne = pygbt_solve_matrix_games(meta_games, method="lcp", mode="one")(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
    nash_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(ne)

    regrets = []
    for i in range(num_players):
        ne_payoff = np.sum(meta_games[i] * nash_prob_matrix)
        dev_prob = ne.copy()
        dev_prob[i] = list(np.append(subgame_ne[i], [0 for _ in range(num_new_pol)]))
        dev_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
        subgame_payoff = np.sum(meta_games[i] * dev_prob_matrix)
        regrets.append(ne_payoff - subgame_payoff)

    return regrets


def sample_episodes(env, agents, number_episodes=1):
    """
    sample pure strategy payoff in an env
    Params:
        agents : a list of length num_player
        env    : open_spiel environment
    Returns:
        a list of length num_player containing players' strategies
    """

    cumulative_rewards = np.zeros(len(agents))

    for _ in range(number_episodes):
        time_step = env.reset()
        cumulative_rewards = 0.0
        while not time_step.last():
            state = env.get_state
            if state.is_simultaneous_node():
                action_list = []
                for agent in agents:
                    output = agent.step(time_step, is_evaluation=True)
                    action_list.append(output.action)
            elif state.is_chance_node():
                outcomes, probs = zip(*env.get_state.chance_outcomes())
                action_list = utils.random_choice(outcomes, probs)
            else:
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
            time_step = env.step(action_list)
            cumulative_rewards += np.array(time_step.rewards)

    return cumulative_rewards / number_episodes


def rollout(env, strategies, strategy_support, sims_per_entry=1000):
    """
    Evaluate player's mixed strategy with support in env.
    Params:
        env              : an open_spiel env
        strategies       : list of list, each list containing a player's strategy
        strategy_support : mixed_strategy support probability vector
        sims_per_entry   : number of episodes for each pure strategy profile to sample
    Return:
        a list of players' payoff
    """
    num_players = len(strategies)
    num_strategies = [len(ele) for ele in strategies]
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(strategy_support)
    payoff_tensor = np.zeros([num_players] + num_strategies)

    for ind in itertools.product(*[np.arange(ele) for ele in num_strategies]):
        strat = [strategies[i][ind[i]] for i in range(num_players)]
        pure_payoff = sample_episodes(env, strat, sims_per_entry)
        payoff_tensor[tuple([...] + list(ind))] = pure_payoff

    return [np.sum(payoff_tensor[i] * prob_matrix) for i in range(num_players)]


class SElogs(object):
    def __init__(self,
                 slow_oracle_period,
                 fast_oracle_period,
                 meta_strategy_methods,
                 heuristic_list):
        self.slow_oracle_period = slow_oracle_period
        self.fast_oracle_period = fast_oracle_period
        self.meta_strategy_methods = meta_strategy_methods
        self.heuristic_list = heuristic_list

        self._slow_oracle_iters = []
        self._fast_oracle_iters = []

        self.regrets = []
        self.nashconv = []

    def update_regrets(self, regrets):
        self.regrets.append(regrets)

    def get_regrets(self):
        return self.regrets

    def update_nashconv(self, nashconv):
        self.nashconv.append(nashconv)

    def get_nashconv(self):
        return self.nashconv

    def update_slow_iters(self, iter):
        self._slow_oracle_iters.append(iter)

    def get_slow_iters(self):
        return self._slow_oracle_iters

    def update_fast_iters(self, iter):
        self._fast_oracle_iters.append(iter)

    def get_fast_iters(self):
        return self._fast_oracle_iters


def smoothing_kl(p, q, eps=0.001):
    p = smoothing(p, eps)
    q = smoothing(q, eps)
    return np.sum(p * np.log(p / q))


def smoothing(p, eps):
    p = np.array(p, dtype=np.float)
    zeros_pos_p = np.where(p == 0)[0]
    num_zeros = len(zeros_pos_p)
    x = eps * num_zeros / (len(p) - num_zeros)
    for i in range(len(p)):
        if i in zeros_pos_p:
            p[i] = eps
        else:
            p[i] -= x
    return p


def isExist(path):
    """
    Check if a path exists.
    :param path: path to check.
    :return: bool
    """
    return os.path.exists(path)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        raise ValueError(path + " already exists.")
    else:
        os.makedirs(path)
        print(path + " has been created successfully.")


def save_pkl(obj, path):
    """
    Pickle a object to path.
    :param obj: object to be pickled.
    :param path: path to save the object
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    """
    Load a pickled object from path
    :param path: path to the pickled object.
    :return: object
    """
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def save_nash(nash_prob, iteration, checkpoint_dir):
    """
    Save nash probabilities
    """
    current_path = os.path.join(checkpoint_dir, 'nash_prob/')
    if not isExist(current_path):
        mkdir(current_path)
    save_pkl(nash_prob, current_path + str(iteration) + '.pkl')


def save_strategies(solver, checkpoint_dir):
    """
    Save all strategies.
    """
    num_players = solver._num_players
    for player in range(num_players):
        current_path = os.path.join(checkpoint_dir, 'strategies/player_' + str(player) + "/")
        if not isExist(current_path):
            mkdir(current_path)
        for i, policy in enumerate(solver.get_policies()[player]):
            if isExist(current_path + str(i + 1) + '.pkl'):
                continue
            save_pkl(policy.get_weights(), current_path + str(i + 1) + '.pkl')



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

def mixed_strategy_payoff_2p(meta_games, probs):
   payoffs = []
   prob1 = probs[0]
   prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
   prob2 = probs[1]
   for meta_game in meta_games:
       payoffs.append(np.sum(prob1 * meta_game * prob2))
   return payoffs

def dev_regret(meta_games, probs):
    """
    Calculate the regret of a profile in an empirical game.
    :param meta_games:
    :param probs:
    :return:
    """
    num_players = 2
    payoffs = mixed_strategy_payoff_2p(meta_games, probs)
    dev_strs, dev_payoff = deviation_strategy(meta_games, probs)
    nashconv = 0
    for player in range(num_players):
        nashconv += np.maximum(dev_payoff[player] - payoffs[player], 0)
    return nashconv

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

    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(probs)
    deviation_payoffs = []

    for i in range(num_players):
        profile_payoff = np.sum(meta_games[i] * prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(num_strategies[i]):
            dev_prob = probs.copy()
            dev_prob[i] = np.zeros(num_strategies[i])
            dev_prob[i][j] = 1
            new_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
            dev_payoff.append(np.sum(meta_games[i] * new_prob_matrix))
        deviation_payoffs.append(dev_payoff - profile_payoff)

    nashconv = np.sum([np.max(ele) for ele in deviation_payoffs])
    return nashconv