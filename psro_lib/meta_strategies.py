import numpy as np

from solution_solvers.nash_solver.projected_replicator_dynamics import projected_replicator_dynamics
from solution_solvers.nash_solver import controled_RD
from solution_solvers.nash_solver.pygambit_solver import pygbt_solve_matrix_games

EPSILON_MIN_POSITIVE_PROBA = 1e-8


def get_joint_strategy_from_marginals(probabilities):
  """Returns a joint strategy matrix from a list of marginals.
  The orginial function does not work with numpy 1.24
  Args:
    probabilities: list of probabilities.

  Returns:
    A joint strategy from a list of marginals.
  """
  joint = np.outer(probabilities[0], probabilities[1])
  for i in range(len(probabilities) - 2):
      joint = joint.reshape(tuple(list(joint.shape) + [1])) * probabilities[i + 2]
  return joint


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


def uniform_strategy(solver, return_joint=False):
  """Returns a Random Uniform distribution on policies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies()
  policy_lengths = [len(pol) for pol in policies]
  result = [np.ones(pol_len) / pol_len for pol_len in policy_lengths]
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def softmax_on_range(number_policies):
  x = np.array(list(range(number_policies)))
  x = np.exp(x-x.max())
  x /= np.sum(x)
  return x


def uniform_biased_strategy(solver, return_joint=False):
  """Returns a Biased Random Uniform distribution on policies.

  The uniform distribution is biased to prioritize playing against more recent
  policies (Policies that were appended to the policy list later in training)
  instead of older ones.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies()
  if not isinstance(policies[0], list):
    policies = [policies]
  policy_lengths = [len(pol) for pol in policies]
  result = [softmax_on_range(pol_len) for pol_len in policy_lengths]
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def renormalize(probabilities):
  """Replaces all negative entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 1e-9] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities



def general_nash_strategy(solver, return_joint=False, mode='one', game=None, checkpoint_dir=None):
    """Returns nash distribution on meta game matrix.
  This method works for general-sum multi-player games.
  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.
    mode: Return one or all or pure NE.
    game: overrides solver.get_meta_games() if provided
  Returns:
    Nash distribution on strategies.
  """
    meta_games = solver.get_meta_game() if game is None else game
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]

    # print(meta_games)
    # Change the "lcp" to "gnm" for games with more than 2 players.
    if len(meta_games) == 2:
        equilibria = pygbt_solve_matrix_games(meta_games, method="lcp", mode=mode)
    else:
        equilibria = pygbt_solve_matrix_games(meta_games, method="gnm", mode=mode)

    # print("NE:", equilibria)
    normalized_eqa = []
    for ne in equilibria:
        normalized_ne = []
        for prob in ne:
            normalized_prob = renormalize(prob)
            normalized_ne.append(normalized_prob)
        normalized_eqa.append(normalized_ne)

    if not return_joint:
        return normalized_eqa[0]
    else:
        if mode == 'all' and type(equilibria[0]) == list:
            # If multiple NE exist, return a list with joint strategies.
            joint_strategies_list = [get_joint_strategy_from_marginals(ne) for ne in normalized_eqa]
            return normalized_eqa, joint_strategies_list
        else:
            joint_strategies = get_joint_strategy_from_marginals(normalized_eqa[0])
            return normalized_eqa[0], joint_strategies


def prd_strategy(solver, return_joint=False):
  """Computes Projected Replicator Dynamics strategies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    PRD-computed strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs()
  result = projected_replicator_dynamics(meta_games, **kwargs)
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies



def regret_controled_RD(solver, return_joint=False, checkpoint_dir=None, regret_threshold=0.50):
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs()
  result = controled_RD.controled_replicator_dynamics(meta_games, regret_threshold=regret_threshold, **kwargs)

  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def self_play_strategy(solver, return_joint=False, checkpoint_dir=None):
    """
    Return a strategy with only the newest strategy in the support (played with probability 1).
    :param solver: GenPSROSolver instance.
    :param return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.
    :return:
    """
    policies = solver.get_policies()
    policy_lengths = [len(pol) for pol in policies]
    result = []
    for pol_len in policy_lengths:
      strategy = np.zeros(pol_len)
      strategy[-1] = 1
      result.append(strategy)
    if not return_joint:
      return result
    else:
      joint_strategies = get_joint_strategy_from_marginals(result)
      return result, joint_strategies


META_STRATEGY_METHODS = {
    "uniform_biased": uniform_biased_strategy,
    "uniform": uniform_strategy,
    "nash": general_nash_strategy,
    "prd": prd_strategy,
    "sp": self_play_strategy,
    "CRD": regret_controled_RD,
}