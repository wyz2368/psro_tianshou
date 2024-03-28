import gymnasium
import numpy as np

from psro_lib.rl_agents.rl_oracle import RLOracle, freeze_all
from psro_lib.rl_agents.mapolicy_trainer_v2 import MAPolicyTrainer_v2
from psro_lib.rl_agents.rl_factory import generate_agent_class
from psro_lib.utils import sample_strategy_marginal
from psro_lib.game_factory import GAMES, get_env_factory

from tianshou.policy import RandomPolicy
from tianshou.policy import MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

# new_env = get_env_factory("tictactoe_v3")()
# new_policy = [RandomPolicy(), RandomPolicy()]
# new_master = MultiAgentPolicyManager(policies=new_policy, env=new_env)
# new_test_envs = DummyVectorEnv([get_env_factory("tictactoe_v3") for _ in range(2)])
# new_test_collector = Collector(new_master, new_test_envs)
# new_collect_result = new_test_collector.collect(n_episode=5)
# print(new_collect_result)

def print_params(policy):
    nn = policy.model
#     print(nn)
    for p in nn.parameters():
        print(p.data)
        break



# env = get_env_factory("tictactoe_v3")()
env = GAMES["tictactoe_v3"]

observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
state_size = observation_space.shape or observation_space.n,
num_actions = env.action_space.shape or env.action_space.n,

agent_kwargs = {
        "state_size": state_size,
        "num_actions": num_actions,
        "hidden_layers_sizes": [128] * 2,
        "batch_size": 32,
        "learning_rate": 1e-2,
        "update_target_network_every": 500,
        "learn_every": 10,
        "optimizer_str": "adam",
        "replay_buffer_size": 20000
    }

agent_class, on_policy = generate_agent_class(agent_name="DQN")


train_envs = DummyVectorEnv([get_env_factory(env.env.metadata["name"])])
test_envs = DummyVectorEnv([get_env_factory(env.env.metadata["name"])])

mapolicy_trainer = MAPolicyTrainer_v2(env=env,
                                      on_policy=on_policy,
                                      best_response_kwargs=agent_kwargs,
                                      train_envs=train_envs,
                                      test_envs=test_envs)

oracle = RLOracle(env=env,
                  trainer=mapolicy_trainer,
                  best_response_class=agent_class,
                  best_response_kwargs=agent_kwargs,
                  number_training_episodes=10,
                  sigma=0.0,
                  verbose=False)


agents = oracle.generate_new_policies()
freeze_all(agents)
total_policies = [[agent, agent] for agent in agents]

print_params(agents[1].model)

# total_policies = [[RandomPolicy()], [RandomPolicy()]]
meta_probabilities = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
strategy_sampler = sample_strategy_marginal

new_policies = oracle(env=env,
                      old_policies=total_policies,
                      meta_probabilities=meta_probabilities,
                      strategy_sampler=strategy_sampler,
                      copy_from_prev=False)

print_params(agents[1].model)
# print(new_policies)



