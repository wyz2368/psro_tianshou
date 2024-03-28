"""Strategy selectors repository."""

import numpy as np

# TODO: Where is this used?
DEFAULT_STRATEGY_SELECTION_METHOD = "probabilistic"

# Constant, specifying the threshold below which probabilities are considered 0.
EPSILON_MIN_POSITIVE_PROBA = 1e-8


def exhaustive(solver, number_policies_selected=1):
  """Returns every player's policies.

  Args:
    solver: A GenPSROSolver instance.
    number_policies_selected: Number of policies to return for each player.
      (Compatibility argument)

  Returns:
    used_policies : List of size 'num_players' of lists of size
      min('number_policies_selected', num_policies') containing selected
      policies.
    used_policies_indexes: List of lists of the same shape as used_policies,
      containing the list indexes of selected policies.
  """
  del number_policies_selected
  policies = solver.get_policies()
  indexes = [list(range(len(pol))) for pol in policies]
  return policies, indexes


# Factory function for more complex filters.
def filter_function_factory(filter_function):
  """Returns a function filtering players' strategies wrt.

  'filter_function'.

  This function is used to select which strategy to start training from. As
  such, and in the Rectified Nash Response logic, filter_function expects a
  certain set of arguments:
    - player_policies: The list of policies for the current player.
    - player: The current player id.
    - effective_number_selected: The effective number of policies to select.
    - solver: In case the above arguments weren't enough, the solver instance so
    the filter_function can have more complex behavior.
  And returns the selected policies and policy indexes for the current player.

  Args:
    filter_function: A filter function following the specifications above, used
      to filter which strategy to start training from for each player.

  Returns:
    A filter function on all players.
  """

  def filter_policies(solver, number_policies_selected=1):
    """Filters each player's policies according to 'filter_function'.

    Args:
      solver: The PSRO solver.
      number_policies_selected: The expected number of policies to select. If
        there are fewer policies than 'number_policies_selected', behavior will
        saturate at num_policies.

    Returns:
      used_policies : List of length 'num_players' of lists of length
        min('number_policies_selected', num_policies') containing selected
        policies.
      used_policies_indexes: List of lists of the same shape as used_policies,
        containing the list indexes of selected policies.

    """
    policies = solver.get_policies()
    num_players = len(policies)
    meta_strategy_probabilities = solver.get_meta_strategies()

    used_policies = []
    used_policy_indexes = []
    for player in range(num_players):
      player_policies = policies[player]
      current_selection_probabilities = meta_strategy_probabilities[player]
      effective_number = min(number_policies_selected, len(player_policies))

      used_policy, used_policy_index = filter_function(
          player_policies, current_selection_probabilities, player,
          effective_number, solver)
      used_policies.append(used_policy)
      used_policy_indexes.append(used_policy_index)
    return used_policies, used_policy_indexes

  # Return the created function.
  return filter_policies


def rectified_filter(player_policies, selection_probabilities, player,
                     effective_number_to_select, solver):
  """Returns every strategy with nonzero selection probability.

  Args:
    player_policies: A list of policies for the current player.
    selection_probabilities: Selection probabilities for 'player_policies'.
    player: Player id.
    effective_number_to_select: Effective number of policies to select.
    solver: PSRO solver instance if kwargs needed.

  Returns:
    selected_policies : List of size 'effective_number_to_select'
      containing selected policies.
    selected_indexes: List of the same shape as selected_policies,
      containing the list indexes of selected policies.
  """
  del effective_number_to_select, solver, player
  selected_indexes = [
      i for i in range(len(player_policies))
      if selection_probabilities[i] > EPSILON_MIN_POSITIVE_PROBA
  ]
  selected_policies = [player_policies[i] for i in selected_indexes]

  return selected_policies, selected_indexes


def probabilistic_filter(player_policies, selection_probabilities, player,
                         effective_number_to_select, solver):
  """Returns every strategy with nonzero selection probability.

  Args:
    player_policies: A list of policies for the current player.
    selection_probabilities: Selection probabilities for 'player_policies'.
    player: Player id.
    effective_number_to_select: Effective number of policies to select.
    solver: PSRO solver instance if kwargs needed.

  Returns:
    selected_policies : List of size 'effective_number_to_select'
      containing selected policies.
    selected_indexes: List of the same shape as selected_policies,
      containing the list indexes of selected policies.
  """
  del solver, player
  selected_indexes = list(
      np.random.choice(
          list(range(len(player_policies))),
          effective_number_to_select,
          replace=False,
          p=selection_probabilities))
  selected_policies = [player_policies[i] for i in selected_indexes]
  return selected_policies, selected_indexes


def top_k_probabilities_filter(player_policies, selection_probabilities, player,
                               effective_number_to_select, solver):
  """Returns top 'effective_number_to_select' highest probability policies.

  Args:
    player_policies: A list of policies for the current player.
    selection_probabilities: Selection probabilities for 'player_policies'.
    player: Player id.
    effective_number_to_select: Effective number of policies to select.
    solver: PSRO solver instance if kwargs needed.

  Returns:
    selected_policies : List of size 'effective_number_to_select'
      containing selected policies.
    selected_indexes: List of the same shape as selected_policies,
      containing the list indexes of selected policies.
  """
  del player, solver
  selected_indexes = [
      index for _, index in sorted(
          zip(selection_probabilities, list(range(len(player_policies)))),
          key=lambda pair: pair[0])
  ][:effective_number_to_select]

  selected_policies = [player_policies[i] for i in selected_indexes]
  return selected_policies, selected_indexes


def uniform_filter(player_policies, selection_probabilities, player,
                   effective_number_to_select, solver):
  """Returns 'effective_number_to_select' uniform-randomly selected policies.

  Args:
    player_policies: A list of policies for the current player.
    selection_probabilities: Selection probabilities for 'player_policies'.
    player: Player id.
    effective_number_to_select: Effective number of policies to select.
    solver: PSRO solver instance if kwargs needed.

  Returns:
    selected_policies : List of size 'effective_number_to_select'
      containing selected policies.
    selected_indexes: List of the same shape as selected_policies,
      containing the list indexes of selected policies.
  """
  del solver, selection_probabilities, player
  selected_indexes = list(
      np.random.choice(
          list(range(len(player_policies))),
          effective_number_to_select,
          replace=False,
          p=np.ones(len(player_policies)) / len(player_policies)))
  selected_policies = [player_policies[i] for i in selected_indexes]
  return selected_policies, selected_indexes


def functional_probabilistic_filter(player_policies, selection_probabilities,
                                    player, effective_number_to_select, solver):
  """Returns effective_number_to_select randomly selected policies by function.

  Args:
    player_policies: A list of policies for the current player.
    selection_probabilities: Selection probabilities for 'player_policies'.
    player: Player id.
    effective_number_to_select: Effective number of policies to select.
    solver: PSRO solver instance if kwargs needed.

  Returns:
    selected_policies : List of size 'effective_number_to_select'
      containing selected policies.
    selected_indexes: List of the same shape as selected_policies,
      containing the list indexes of selected policies.
  """
  kwargs = solver.get_kwargs()
  # By default, use meta strategies.
  probability_computation_function = kwargs.get(
      "selection_probability_function") or (lambda x: x.get_meta_strategies())

  selection_probabilities = probability_computation_function(solver)[player]
  selected_indexes = list(
      np.random.choice(
          list(range(len(player_policies))),
          effective_number_to_select,
          replace=False,
          p=selection_probabilities))
  selected_policies = [player_policies[i] for i in selected_indexes]
  return selected_policies, selected_indexes


def get_current_and_average_payoffs(ps2ro_trainer, current_player,
                                    current_strategy):
  """Returns the current player's and average players' payoffs.

  These payoffs are returned when current_player's strategy's index is
  'current_strategy'.

  Args:
    ps2ro_trainer: A ps2ro object.
    current_player: Integer, current player index.
    current_strategy: Integer, current player's strategy index.

  Returns:
    Payoff tensor for current player, Average payoff tensor over all players.
  """
  # Get the vector of payoffs associated with current_player's strategy ind
  meta_games = ps2ro_trainer.meta_games
  current_payoff = meta_games[current_player]
  current_payoff = np.take(
      current_payoff, current_strategy, axis=current_player)

  # Get average per-player payoff matrix.
  average_payoffs = np.mean(meta_games, axis=0)
  average_payoffs = np.take(
      average_payoffs, current_strategy, axis=current_player)
  return current_payoff, average_payoffs

def rectified_selector(ps2ro_trainer, current_player, current_strategy):
  current_payoff, average_payoffs = get_current_and_average_payoffs(
      ps2ro_trainer, current_player, current_strategy)

  # Rectified Nash condition : select those strategies where we do better
  # than others.
  res = current_payoff >= average_payoffs
  return np.expand_dims(res, axis=current_player)



# Introducing aliases:
uniform = filter_function_factory(uniform_filter)
rectified = filter_function_factory(rectified_filter)
probabilistic = filter_function_factory(probabilistic_filter)
top_k_probabilities = filter_function_factory(top_k_probabilities_filter)
functional_probabilistic = filter_function_factory(
    functional_probabilistic_filter)


TRAINING_STRATEGY_SELECTORS = {
    "probabilistic": probabilistic,
    "rectified": rectified
}
