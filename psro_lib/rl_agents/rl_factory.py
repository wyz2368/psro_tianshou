from tianshou.policy import DQNPolicy, PPOPolicy, PGPolicy, RandomPolicy

def generate_agent_class(agent_name:str):
    if agent_name == "PG":
      return PGPolicy, True # on_policy: True
    elif agent_name == "DQN":
      return DQNPolicy, False
    elif agent_name == "PPO":
      return PPOPolicy, True
    elif agent_name == "random":
      return RandomPolicy, None
    else:
      raise NotImplementedError("The available oracle classes are PG, DQN, PPO, random")
