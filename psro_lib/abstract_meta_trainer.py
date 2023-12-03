import numpy as np
from psro_lib import meta_strategies
from psro_lib import strategy_selectors
from psro_lib.utils import init_logger
from psro_lib.rl_agents.utils import get_env_factory
from psro_lib.rl_agents.master_policy import MultiAgentPolicyManager_PSRO

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager




_DEFAULT_STRATEGY_SELECTION_METHOD = "probabilistic"
_DEFAULT_META_STRATEGY_METHOD = "prd"


def _process_string_or_callable(string_or_callable, dictionary):
  """Process a callable or a string representing a callable.

  Args:
    string_or_callable: Either a string or a callable
    dictionary: Dictionary of shape {string_reference: callable}

  Returns:
    string_or_callable if string_or_callable is a callable ; otherwise,
    dictionary[string_or_callable]

  Raises:
    NotImplementedError: If string_or_callable is of the wrong type, or has an
      unexpected value (Not present in dictionary).
  """
  if callable(string_or_callable):
    return string_or_callable

  try:
    return dictionary[string_or_callable]
  except KeyError as e:
    raise NotImplementedError("Input type / value not supported. Accepted types"
                              ": string, callable. Acceptable string values : "
                              "{}. Input provided : {}".format(
                                  list(dictionary.keys()),
                                  string_or_callable)) from e



class AbstractMetaTrainer(object):
  """Abstract class implementing meta trainers.

  If a trainer is something that computes a best response to given environment &
  agents, a meta trainer will compute which best responses to compute (Against
  what, how, etc)
  This class can support PBT, Hyperparameter Evolution, etc.
  """

  # pylint:disable=dangerous-default-value
  def __init__(self,
               game,
               oracle,
               initial_policies=None,
               meta_strategy_method=_DEFAULT_META_STRATEGY_METHOD,
               training_strategy_selector=_DEFAULT_STRATEGY_SELECTION_METHOD,
               symmetric_game=False,
               number_policies_selected=1,
               checkpoint_dir=None,
               dummy_env=True,
               **kwargs):
    """Abstract Initialization for meta trainers.

    Args:
      game: A PettingZoo Wrapper object.
      oracle: An oracle object.
      initial_policies: A list of initial policies, to set up a default for
        training. Resorts to tabular policies if not set.
      meta_strategy_method: String, or callable taking a MetaTrainer object and
        returning a list of meta strategies (One list entry per player).
        String value can be:
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      training_strategy_selector: A callable or a string. If a callable, takes
        as arguments: - An instance of `PSROSolver`, - a
          `number_policies_selected` integer. and returning a list of
          `num_players` lists of selected policies to train from.
        When a string, supported values are:
              - "probabilistic": randomly selects 'number_policies_selected'
                with probabilities determined by the meta strategies.
      symmetric_game: Whether to consider the current game as symmetric (True)
        game or not (False).
      number_policies_selected: Maximum number of new policies to train for each
        player at each PSRO iteration.
      **kwargs: kwargs for meta strategy computation and training strategy
        selection
    """
    self._iterations = 0
    self._game = game
    self._oracle = oracle
    self._num_players = len(self._game.agents)
    self._game_num_players = self._num_players

    # Tianshou
    self.logger = init_logger(logger_name=__name__, checkpoint_dir=checkpoint_dir)
    self.dummy_env = dummy_env
    env_name = self._game.env.metadata["name"]
    if self.dummy_env:
      self.test_envs = DummyVectorEnv([get_env_factory(env_name)])
    else:
      self.test_envs = SubprocVectorEnv([get_env_factory(env_name) for _ in range(5)])


    self.symmetric_game = symmetric_game
    self._num_players = 1 if symmetric_game else self._num_players

    self._number_policies_selected = number_policies_selected

    meta_strategy_method = _process_string_or_callable(
        meta_strategy_method, meta_strategies.META_STRATEGY_METHODS)

    self._training_strategy_selector = _process_string_or_callable(
        training_strategy_selector,
        strategy_selectors.TRAINING_STRATEGY_SELECTORS)

    self._meta_strategy_method = meta_strategy_method
    self._kwargs = kwargs

    self._initialize_policy(initial_policies)
    self._initialize_game_state()
    self.update_meta_strategies()

  def _initialize_policy(self, initial_policies):
    return NotImplementedError(
        "initialize_policy not implemented. Initial policies passed as"
        " arguments : {}".format(initial_policies))

  def _initialize_game_state(self):
    return NotImplementedError("initialize_game_state not implemented.")

  def iteration(self, seed=None):
    """Main trainer loop.

    Args:
      seed: Seed for random BR noise generation.
    """
    self._iterations += 1
    self.update_agents()  # Generate new, Best Response agents via oracle.
    self.update_empirical_gamestate(seed=seed)  # Update gamestate matrix.
    self.update_meta_strategies()  # Compute meta strategy (e.g. Nash)

  def update_meta_strategies(self):
    self._meta_strategy_probabilities = self._meta_strategy_method(self)
    if self.symmetric_game:
      self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

  def update_agents(self):
    return NotImplementedError("update_agents not implemented.")

  def update_empirical_gamestate(self, seed=None):
    return NotImplementedError("update_empirical_gamestate not implemented."
                               " Seed passed as argument : {}".format(seed))

  def sample_episodes(self, policies, num_episodes): #TODO: Test this function
    """Samples episodes and averages their returns.

    Args:
      policies: A list of policies representing the policies executed by each
        player.
      num_episodes: Number of episodes to execute to estimate average return of
        policies.

    Returns:
      Average episode return over num episodes.
    """
    master_policy = MultiAgentPolicyManager(policies=policies, env=self._game)
    self.test_envs.reset()
    test_collector = Collector(master_policy, self.test_envs)
    collect_result = test_collector.collect(n_episode=num_episodes)
    return collect_result["rews"].mean(axis=0)

  #TODO: check collector behavior for multi-agent.

  def get_meta_strategies(self):
    """Returns the Nash Equilibrium distribution on meta game matrix."""
    meta_strategy_probabilities = self._meta_strategy_probabilities
    if self.symmetric_game:
      meta_strategy_probabilities = (self._game_num_players *
                                     meta_strategy_probabilities)
    return [np.copy(a) for a in meta_strategy_probabilities]

  def get_meta_game(self):
    """Returns the meta game matrix."""
    meta_games = self._meta_games
    return [np.copy(a) for a in meta_games]

  def get_policies(self):
    """Returns the players' policies."""
    policies = self._policies
    if self.symmetric_game:
      policies = self._game_num_players * policies
    return policies

  def get_kwargs(self):
    return self._kwargs

  ## From Yongzhao's Repo##
  def update_meta_strategy_method(self, new_meta_str_method=None):
    """
    Update meta-strategy method and corresponding name.
    :param new_meta_str_method: new meta-strategy method.
    :return:
    """
    if new_meta_str_method is not None:
      # meta_strategy alias and name not corrherent
      if '_strategy' in new_meta_str_method:
        new_meta_str_method = new_meta_str_method[:new_meta_str_method.index('_strategy')]
      self._meta_strategy_method = _process_string_or_callable(new_meta_str_method, meta_strategies.META_STRATEGY_METHODS)
      self.logger.info("Using {} as strategy method.".format(self._meta_strategy_method.__name__))
      self._meta_strategy_method_name = self._meta_strategy_method.__name__
      self.update_meta_strategies()  # Compute meta strategy (e.g. Nash)

      self.update_meta_strategies()

  def get_meta_strategy_method(self):
    """
    Return the name and the function of current meta-strategy method.
    :return:
    """
    return self._meta_strategy_method_name, self._meta_strategy_method

  def get_nash_strategies(self):
    """Returns the nash meta-strategy distribution on meta game matrix. When other meta strategies in play, nash strategy is still needed for evaluation
    """
    meta_strategy_probabilities = meta_strategies.general_nash_strategy(self)
    return [np.copy(a) for a in meta_strategy_probabilities]

  def get_prd_strategies(self):
    meta_strategy_probabilities = meta_strategies.prd_strategy(self)
    return [np.copy(a) for a in meta_strategy_probabilities]

  def get_crd_strategies(self):
    meta_strategy_probabilities = meta_strategies.regret_controled_RD(self)
    return [np.copy(a) for a in meta_strategy_probabilities]

  def get_uniform_strategies(self):
    meta_strategy_probabilities = meta_strategies.uniform_strategy(self)
    return [np.copy(a) for a in meta_strategy_probabilities]

  def update_policy_in_test_collector(self, test_collector, new_master_policy):
    test_collector.policy = new_master_policy