import pyspiel
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
from tianshou.env.pettingzoo_env import PettingZooEnv
from psro_lib.psro_example import get_space_sizes

def run_env_from_spiel(game_name="2048"):
    env = pyspiel.load_game(game_name)
    env = OpenSpielCompatibilityV0(env)

    # print(get_space_sizes(env))

    env = PettingZooEnv(env)
    print(env.agent_idx)

    # print("meta_data:", env.game_name)
    # print("possible_agents:", env.possible_agents)
    # print("observation_spaces:", env.observation_spaces)
    # print("action_spaces:", env.action_spaces)

    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()
    #     # print(observation)
    #     # print(reward)
    #     # print(termination)
    #     # print(truncation)
    #     if termination or truncation:
    #         action = None
    #     else:
    #         action = env.action_space(agent).sample(info["action_mask"])  # this is where you would insert your policy
    #     env.step(action)



if __name__ == "__main__":
    # Games:
    # kuhn_poker
    # leduc_poker
    # blotto
    run_env_from_spiel("kuhn_poker")