"""
An implementations of the PSRO trainer class inherited from abstract_meta_trainer.
"""

import itertools

import numpy as np
from psro_lib import abstract_meta_trainer
from psro_lib import strategy_selectors
from psro_lib import utils
# from psro_lib.rl_agents.rl_factory import RandomPolicy
from tianshou.policy import RandomPolicy

# This allows implementation of rectified Nash.
TRAIN_TARGET_SELECTORS = {
    "": None,
    "rectified": strategy_selectors.rectified_selector,
}

class PSROSolver(abstract_meta_trainer.AbstractMetaTrainer):
  """
  A general implementation PSRO.
  """

  def __init__(self,
               game,
               oracle,
               sims_per_entry,
               initial_policies=None,
               rectifier="",
               meta_strategy_method="prd",
               sample_from_marginals=False,
               **kwargs):
    """Initialize the PSRO solver.

    Arguments:
      game: The envionrment.
      oracle: Callable that takes as input: - game - policy - policies played -
        array representing the probability of playing policy i - other kwargs
        and returns a new best response.
      sims_per_entry: Number of simulations to run to estimate each element of
        the game outcome matrix.
      initial_policies: A list of initial policies for each player, from which
        the optimization process will start.
      rectifier: A string indicating the rectifying method. Can be :
              - "" or None: Train against potentially all strategies.
              - "rectified": Train only against strategies beaten by current
                strategy.
      meta_strategy_method: String or callable taking a GenPSROSolver object and
        returning two lists ; one list of meta strategies (One list entry per
        player), and one list of joint strategies.
        String value can be:
              - alpharank: AlphaRank distribution on policies.
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      sample_from_marginals: A boolean, specifying whether to sample from
        marginal (True) or joint (False) meta-strategy distributions.
      **kwargs: kwargs for meta strategy computation and training strategy
        selection.
    """

    self._sims_per_entry = sims_per_entry
    self._rectifier = TRAIN_TARGET_SELECTORS.get(
        rectifier, None)
    self._rectify_training = self._rectifier

    self._meta_strategy_probabilities = np.array([])
    self._non_marginalized_probabilities = np.array([])

    self._policies = []  # A list of size `num_players` of lists containing the strategies of each player.
    self._new_policies = []

    self.sample_from_marginals = sample_from_marginals

    super(PSROSolver, self).__init__(
        game,
        oracle,
        initial_policies,
        meta_strategy_method,
        **kwargs)

  def _initialize_policy(self, initial_policies):
    """
    Initialize some strategies for PSRO. If initial_policies is not given,
    then fill with random strategies.
    """
    if self.symmetric_game:
      self._policies = [[]]
      self._new_policies = [
          (
              [initial_policies[0]]
              if initial_policies
    else [RandomPolicy()] #TODO: Check if random policy needs input. Should it match a particular game.
          )
      ]
    else:
      self._policies = [[] for _ in range(self._num_players)]
      self._new_policies = [
          (
              [initial_policies[k]]
              if initial_policies
              else [RandomPolicy()] #TODO: Check if random policy needs input.
          )
          for k in range(self._num_players)
      ]

  def _initialize_game_state(self):
    """
    Initialize the empirical game, simulting the profiles given by initial strategies.
    The empirical game is a list of payoff matrix represented by numpy array, one for each player.
    """
    effective_payoff_size = self._game_num_players
    self._meta_games = [
        np.array(utils.empty_list_generator(effective_payoff_size))
        for _ in range(effective_payoff_size)
    ]
    self.update_empirical_gamestate(seed=None)

  def update_meta_strategies(self):
    """Recomputes the current meta strategy (best response target) of each player.

    Given new payoff tables, we call self._meta_strategy_method to update the
    meta-probabilities.
    """
    if self.symmetric_game:
      self._policies = self._policies * self._game_num_players

    self._meta_strategy_probabilities, self._non_marginalized_probabilities = (
        self._meta_strategy_method(solver=self, return_joint=True))

    if self.symmetric_game:
      self._policies = [self._policies[0]]
      self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

  def get_policies_and_strategies(self):
    """Returns current policy sampler, policies and meta-strategies of the game.

    If strategies are rectified, we automatically switch to returning joint
    strategies.

    Returns:
      sample_strategy: A strategy sampling function
      total_policies: A list of list of policies, one list per player.
      probabilities_of_playing_policies: the meta strategies, either joint or
        marginalized.
    """
    sample_strategy = utils.sample_strategy_marginal
    probabilities_of_playing_policies = self.get_meta_strategies()
    if self._rectify_training or not self.sample_from_marginals:
      sample_strategy = utils.sample_strategy_joint
      probabilities_of_playing_policies = self._non_marginalized_probabilities

    total_policies = self.get_policies()
    return sample_strategy, total_policies, probabilities_of_playing_policies

  def _restrict_target_training(self,
                                current_player,
                                ind,
                                total_policies,
                                probabilities_of_playing_policies,
                                restrict_target_training_bool,
                                epsilon=1e-12):
    """Rectifies training (Unused).

    Args:
      current_player: the current player.
      ind: Current strategy index of the player.
      total_policies: all policies available to all players.
      probabilities_of_playing_policies: meta strategies.
      restrict_target_training_bool: Boolean specifying whether to restrict
        training. If False, standard meta strategies are returned. Otherwise,
        restricted joint strategies are returned.
      epsilon: threshold below which we consider 0 sum of probabilities.

    Returns:
      Probabilities of playing each joint strategy (If rectifying) / probability
      of each player playing each strategy (Otherwise - marginal probabilities)
    """
    true_shape = tuple([len(a) for a in total_policies])
    if not restrict_target_training_bool:
      return probabilities_of_playing_policies
    else:
      kept_probas = self._rectifier(self, current_player, ind)
      # Ensure probabilities_of_playing_policies has same shape as kept_probas.
      probability = probabilities_of_playing_policies.reshape(true_shape)
      probability = probability * kept_probas
      prob_sum = np.sum(probability)

      # If the rectified probabilities are too low / 0, we play against the
      # non-rectified probabilities.
      if prob_sum <= epsilon:
        probability = probabilities_of_playing_policies
      else:
        probability /= prob_sum

      return probability

  def update_agents(self):
    """
    Updates policies for each player at the same time by calling the oracle.
    The resulting policies are appended to self._new_policies.
    (This function is significantly different from the original.)
    """

    (sample_strategy,
     total_policies,
     probabilities_of_playing_policies) = self.get_policies_and_strategies()

    if self.symmetric_game:
        # Notice that the following line returns N references to the same policy
        # This might not be correct for certain applications.
        # E.g., a DQN BR oracle with player_id information
        self._policies = self._game_num_players * self._policies
        self._num_players = self._game_num_players

    # List of List of new policies (One list per player) #TODO: check the oracle.
    # ===================================
    # The oracle is given by Tianshou.
    # ===================================
    self._new_policies = self._oracle(env=self._game,
                                      old_policies=total_policies,
                                      meta_probabilities=probabilities_of_playing_policies,
                                      strategy_sampler=sample_strategy,
                                      copy_from_prev=False)
    # This is the form of self._new_policies.
    # self._new_policies = [[RandomPolicy()], [RandomPolicy()]]

    if self.symmetric_game:
      self._policies = [self._policies[0]]
      self._num_players = 1

  def update_empirical_gamestate(self, seed=None):
    """Given new agents in _new_policies, update meta_games through simulations.

    Args:
      seed: Seed for environment generation.

    Returns:
      Meta game payoff matrix.
    """
    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracle is not None

    if self.symmetric_game:
      # Switch to considering the game as a symmetric game where players have
      # the same policies & new policies. This allows the empirical gamestate
      # update to function normally.
      self._policies = self._game_num_players * self._policies
      self._new_policies = self._game_num_players * self._new_policies
      self._num_players = self._game_num_players

    # Concatenate both lists.
    updated_policies = [
        self._policies[k] + self._new_policies[k]
        for k in range(self._num_players)
    ]

    # Each metagame will be (num_strategies)^self._num_players.
    # There are self._num_player metagames, one per player.
    total_number_policies = [
        len(updated_policies[k]) for k in range(self._num_players)
    ]
    number_older_policies = [
        len(self._policies[k]) for k in range(self._num_players)
    ]
    number_new_policies = [
        len(self._new_policies[k]) for k in range(self._num_players)
    ]

    # Initializing the matrix with nans to recognize unestimated states.
    meta_games = [
        np.full(tuple(total_number_policies), np.nan)
        for k in range(self._num_players)
    ]

    # Filling the matrix with already-known values.
    older_policies_slice = tuple(
        [slice(len(self._policies[k])) for k in range(self._num_players)])
    for k in range(self._num_players):
      meta_games[k][older_policies_slice] = self._meta_games[k]

    # Filling the matrix for newly added policies.
    for current_player in range(self._num_players):
      # Only iterate over new policies for current player ; compute on every
      # policy for the other players.
      range_iterators = [
          range(total_number_policies[k]) for k in range(current_player)
      ] + [range(number_new_policies[current_player])] + [
          range(total_number_policies[k])
          for k in range(current_player + 1, self._num_players)
      ]
      for current_index in itertools.product(*range_iterators):
        used_index = list(current_index)
        used_index[current_player] += number_older_policies[current_player]
        if np.isnan(meta_games[current_player][tuple(used_index)]):
          # estimated_policies is a profile in the empirical game awaiting for evaluation.
          estimated_policies = [
              updated_policies[k][current_index[k]]
              for k in range(current_player)
          ] + [
              self._new_policies[current_player][current_index[current_player]]
          ] + [
              updated_policies[k][current_index[k]]
              for k in range(current_player + 1, self._num_players)
          ]

          if self.symmetric_game:
            # samples to estimate each payoff table entry. This should be
            # brought to sims_per_entry to coincide with expected behavior.

            utility_estimates = self.sample_episodes(estimated_policies,
                                                     self._sims_per_entry)

            player_permutations = list(itertools.permutations(list(range(
                self._num_players))))
            for permutation in player_permutations:
              used_tuple = tuple([used_index[i] for i in permutation])
              for player in range(self._num_players):
                if np.isnan(meta_games[player][used_tuple]):
                  meta_games[player][used_tuple] = 0.0
                meta_games[player][used_tuple] += utility_estimates[
                    permutation[player]] / len(player_permutations)
          else:
            utility_estimates = self.sample_episodes(estimated_policies,
                                                     self._sims_per_entry)
            for k in range(self._num_players):
              meta_games[k][tuple(used_index)] = utility_estimates[k]

    if self.symmetric_game:
      # Make PSRO consider that we only have one population again, as we
      # consider that we are in a symmetric game (No difference between players)
      self._policies = [self._policies[0]]
      self._new_policies = [self._new_policies[0]]
      updated_policies = [updated_policies[0]]
      self._num_players = 1

    self._meta_games = meta_games
    self._policies = updated_policies
    return meta_games

  def get_meta_game(self):
    """Returns the meta game matrix."""
    return self._meta_games

  @property
  def meta_games(self):
    return self._meta_games

  def get_policies(self):
    """Returns a list, each element being a list of each player's policies."""
    policies = self._policies
    if self.symmetric_game:
      policies = self._game_num_players * self._policies
    return policies
