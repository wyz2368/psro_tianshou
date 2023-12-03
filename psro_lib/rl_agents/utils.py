from tianshou.env.pettingzoo_env import PettingZooEnv
from psro_lib.game_factory import games


def get_env_factory(game_name):
    def get_env():
        # env = PettingZooEnv(games[game_name].env())
        env = games[game_name]
        return env
    return get_env