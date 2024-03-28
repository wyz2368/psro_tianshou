"""
Example running PSRO.
"""

import time
import datetime
import os
from absl import app
from absl import flags
import numpy as np
import pickle
import random
from tensorboardX import SummaryWriter

from psro_lib.psro import PSROSolver
from psro_lib.game_factory import GAMES, get_env_factory
from psro_lib.rl_agents.rl_factory import generate_agent_class
from psro_lib.rl_agents.rl_oracle import RLOracle, freeze_all
from psro_lib.rl_agents.mapolicy_trainer_v2 import MAPolicyTrainer_v2
from psro_lib.utils import init_logger
from psro_lib.rl_agents.rl_oracle import print_params

# Tianshou
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import RandomPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
import gymnasium


FLAGS = flags.FLAGS
# Game-related
flags.DEFINE_string("game_name", "kuhn_poker", "Game names: kuhn_poker, tictactoe_v3, leduc_holdem_v4")
flags.DEFINE_integer("n_players", 2, "The number of players.")

# PSRO related
flags.DEFINE_string("meta_strategy_method", "nash",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("sims_per_entry", 10,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 10,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                                           "game as a symmetric game.")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "(No filtering), 'rectified' for rectified.")


# General (RL) agent parameters
#TODO: make consistent with tianshou
flags.DEFINE_string("oracle_type", "DQN", "Choices are DQN, PG (Policy "
                                          "Gradient), PPO, random")
flags.DEFINE_integer("number_training_episodes", int(500), "Number training (default 1e4) " ############
                                                           "episodes per RL policy. Used for PG and DQN")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size")
flags.DEFINE_integer("hidden_layers", 4, "Hidden layer size")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")
flags.DEFINE_integer("n_hidden_layers", 4, "# of hidden layers")

# Policy Gradient Oracle related
flags.DEFINE_string("loss_str", "qpg", "Name of loss used for BR training.")
flags.DEFINE_integer("num_q_before_pi", 8, "# critic updates before Pi update")
flags.DEFINE_float("entropy_cost", 0.001, "Self play proportion")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

# DQN
flags.DEFINE_float("dqn_learning_rate", 5e-2, "DQN learning rate.")
flags.DEFINE_integer("update_target_network_every", 500, "Update target "
                                                         "network every [X] steps")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")
flags.DEFINE_integer("replay_buffer_size", 10000, "replay_buffer_size")
flags.DEFINE_integer("num_parallel_envs", 5, "The number of parallel envs for VectorEnv")
flags.DEFINE_integer("min_buffer_size_to_learn", 100, "min_buffer_size_to_learn (default 1000)") #########

# Oracle
flags.DEFINE_integer("epsilon_decay_duration", int(400), "epsilon_decay_duration.") ############
flags.DEFINE_integer("estimation_step", 1, "The number of steps to look ahead.")
flags.DEFINE_float("discount_factor", 1.0, "discount factor.")
flags.DEFINE_float("epsilon_start", 1.0, "epsilon exploration rate start.")
flags.DEFINE_float("epsilon_end", 0.1, "epsilon exploration rate end.")

# General
flags.DEFINE_string("root_result_folder", 'root_result_psro', "root directory of saved results")
flags.DEFINE_integer("seed", None, "Seed.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
flags.DEFINE_bool("dummy_env", True, "Enables dummy env otherwise subproc env")


def get_space_sizes(env):
    # Assume the players' observation space shapes are same.
    if isinstance(env, OpenSpielCompatibilityV0):
        observation_space = env.observation_spaces["player_0"]
        action_space = env.action_spaces["player_0"]
    else:
        observation_space = (
            env.observation_space["observation"]
            if isinstance(env.observation_space, gymnasium.spaces.Dict)
            else env.observation_space
        )
        action_space = env.action_space
    state_size = observation_space.shape or observation_space.n
    num_actions = action_space.shape or action_space.n

    return state_size, num_actions


#TODO: Extend this to multiple RL algorithms.
def init_dqn_responder(env):
    state_size, num_actions = get_space_sizes(env)

    # Return the RL policy from Tianshou. Currently, assume DQN.
    agent_class, on_policy = generate_agent_class(agent_name=FLAGS.oracle_type)

    # TODO: Check this consistency.
    agent_kwargs = {
        "state_size": state_size,
        "num_actions": num_actions,
        "hidden_layers_sizes": FLAGS.hidden_layer_size,
        "hidden_layers": FLAGS.hidden_layers,
        "batch_size": FLAGS.batch_size,
        "learning_rate": FLAGS.dqn_learning_rate,
        "update_target_network_every": FLAGS.update_target_network_every,
        "learn_every": FLAGS.learn_every,
        "optimizer_str": FLAGS.optimizer_str,
        "replay_buffer_size": FLAGS.replay_buffer_size,
        "num_parallel_envs": FLAGS.num_parallel_envs,
        "discount_factor": FLAGS.discount_factor,
        "estimation_step": FLAGS.estimation_step,
        "epsilon_start": FLAGS.epsilon_start,
        "epsilon_end": FLAGS.epsilon_end,
        "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
        "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn
    }

    train_envs = DummyVectorEnv([get_env_factory(FLAGS.game_name)])
    if FLAGS.dummy_env:
        test_envs = DummyVectorEnv([get_env_factory(FLAGS.game_name)])
    else:
        test_envs = SubprocVectorEnv([get_env_factory(FLAGS.game_name) for _ in range(FLAGS.num_parallel_envs)])

    # Create an MAPolicyTrainer.
    mapolicy_trainer = MAPolicyTrainer_v2(env=env,
                                          on_policy=on_policy,
                                          best_response_kwargs=agent_kwargs,
                                          train_envs=train_envs,
                                          test_envs=test_envs)

    # Create BR oracle.
    oracle = RLOracle(env=env,
                      trainer=mapolicy_trainer,
                      best_response_class=agent_class,
                      best_response_kwargs=agent_kwargs,
                      number_training_episodes=FLAGS.number_training_episodes,
                      sigma=FLAGS.sigma,
                      verbose=FLAGS.verbose)

    agents = oracle.generate_new_policies()
    freeze_all(agents)

    # agents = [RandomPolicy(), RandomPolicy()]

    return oracle, agents


def save_at_termination(solver, file_for_meta_game):
    with open(file_for_meta_game, 'wb') as f:
        pickle.dump(solver.get_meta_game(), f)


def gpsro_looper(env, oracle, agents, writer, checkpoint_dir=None, seed=None):
    """Initializes and executes the GPSRO training loop."""
    sample_from_marginals = True

    # Logging important information
    logger = init_logger(logger_name=__name__, checkpoint_dir=checkpoint_dir)
    logger.info("Game name: {}".format(FLAGS.game_name))
    logger.info("Number of players: {}".format(FLAGS.n_players))
    logger.info("Meta strategy method: {}".format(FLAGS.meta_strategy_method))
    logger.info("Oracle type: {}".format(FLAGS.oracle_type))

    # Create a PSRO solver.
    g_psro_solver = PSROSolver(env,
                            oracle,
                            initial_policies=agents,
                            rectifier=FLAGS.rectifier,
                            sims_per_entry=FLAGS.sims_per_entry,
                            meta_strategy_method=FLAGS.meta_strategy_method,
                            prd_iterations=int(1e5),  # 50000
                            prd_gamma=1e-6,
                            sample_from_marginals=sample_from_marginals,
                            symmetric_game=FLAGS.symmetric_game,
                            checkpoint_dir=checkpoint_dir,
                            dummy_env=FLAGS.dummy_env)

    start_time = time.time()
    for gpsro_iteration in range(1, FLAGS.gpsro_iterations + 1):
        if FLAGS.verbose:
            logger.info("\n===========================\n")
            logger.info("Iteration : {}".format(gpsro_iteration))
            logger.info("Time so far: {}".format(time.time() - start_time))
        g_psro_solver.iteration(seed=seed)
        meta_game = g_psro_solver.get_meta_game()
        meta_probabilities = g_psro_solver.get_meta_strategies()
        nash_meta_probabilities = g_psro_solver.get_nash_strategies()

        if FLAGS.verbose:
            logger.info("Meta game : {}".format(meta_game))
            logger.info("Probabilities : {}".format(meta_probabilities))
            logger.info("Nash Probabilities : {}".format(nash_meta_probabilities))

        if gpsro_iteration == FLAGS.gpsro_iterations:
            save_at_termination(solver=g_psro_solver, file_for_meta_game=checkpoint_dir + '/meta_game.pkl')

        #TODO: Add measure like regret.

    # policies = g_psro_solver.get_policies()
    # print("main:", policies[0][0] is policies[1][0])
    # for policy in policies[0]:
    #     print_params(policy)
    # print("----1----")
    # for policy in policies[1]:
    #     print_params(policy)





def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.seed is None:
        seed = np.random.randint(low=0, high=1e5)
    else:
        seed = FLAGS.seed
    np.random.seed(seed)
    random.seed(seed)

    # Load game. This should be adaptive to different environments.
    env = get_env_factory(FLAGS.game_name)()

    # Set up working directory.
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)

    checkpoint_dir = FLAGS.game_name
    checkpoint_dir = checkpoint_dir + "_oracle_" + FLAGS.oracle_type + '_se_' + str(seed) + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    writer = SummaryWriter(logdir=checkpoint_dir + '/log')

    # Initialize oracle and agents
    if FLAGS.oracle_type == "DQN":
        oracle, agents = init_dqn_responder(env=env)
    else:
        raise ValueError("No such oracle type.")


    gpsro_looper(env, oracle, agents, writer, checkpoint_dir=checkpoint_dir, seed=seed)

    writer.close()


if __name__ == "__main__":
    app.run(main)
