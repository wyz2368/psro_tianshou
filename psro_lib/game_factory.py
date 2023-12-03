from shimmy import OpenSpielCompatibilityV0
from tianshou.env.pettingzoo_env import PettingZooEnv
from pettingzoo.classic import tictactoe_v3
from pettingzoo.classic import leduc_holdem_v4

games = {
    "tictactoe_v3": PettingZooEnv(tictactoe_v3.env()),
    "leduc_holdem_v4": PettingZooEnv(leduc_holdem_v4.env())
}
