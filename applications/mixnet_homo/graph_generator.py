import numpy as np
import random


class HOMOGraph():
    def __init__(self,
                 num_layers, # Number of layers
                 num_nodes_per_layer, # list of Number of nodes per layer
                 false_alarm, # List of false alarm rate
                 false_negative, # List of false negative rate
                 a_attack_cost, # List of attacker's attacking cost
                 a_deploy_cost, # List of attacker's deployment cost
                 a_maintain_cost, # List of attacker's maintaining cost
                 active_rate, # List of the possibility of sucessfully activate a node
                 d_exclude_cost, # List of defender's cost on excluding a node
                 d_deploy_cost, # List of defender's deployment cost
                 d_maintain_cost, # List of defender's maintaining cost
                 usage_threshold=0.03, # the lower bound of usage without penalty
                 d_penalty=-50, # defender's penalty for insufficient usage
                 a_alpha=500,  # coefficient for the reward
                 d_beta=1000, # coefficient for the penalty of attacks.
                 normal_nodes=None, # list of ratio of normal nodes
                 compromised_nodes=None,
                 seed=None): # list of ratio of compromised nodes

        self.num_layers = num_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        self.false_alarm = false_alarm
        self.false_negative = false_negative
        self.a_attack_cost = a_attack_cost
        self.a_deploy_cost = a_deploy_cost
        self.a_maintain_cost = a_maintain_cost
        self.a_alpha = a_alpha
        self.active_rate = active_rate
        self.d_exclude_cost = d_exclude_cost
        self.d_deploy_cost = d_deploy_cost
        self.d_maintain_cost = d_maintain_cost
        self.usage_threshold = usage_threshold
        self.d_penalty = d_penalty
        self.d_beta = d_beta

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if normal_nodes is not None:
            self.normal_nodes = normal_nodes
        else:
            # Uniformly sample the ratio of normal nodes.
            self.normal_nodes = np.random.uniform(0.4, 1, num_layers)

        if compromised_nodes is not None:
            self.compromised_nodes = compromised_nodes
        else:
            # Uniformly sample the ratio of comprehensive nodes.
            self.compromised_nodes = np.zeros(num_layers)
            for layer in range(num_layers):
                self.compromised_nodes[layer] = np.random.uniform(low=0, high=1-self.normal_nodes[layer])

        self.open_nodes = 1 - self.normal_nodes - self.compromised_nodes

        # The ratio of nodes belongs to either players.
        self.att_deploy = np.zeros(self.num_layers)
        self.def_deploy = self.normal_nodes + self.compromised_nodes


    def step(self, def_actions, att_actions):
        """
        step function.
        :param def_actions: [attack proportion * num_layers deploy proportion * num_layers]
        :param att_actions: [exclude proportion * num_layers deploy proportion * num_layers]
        """
        if not isinstance(def_actions, np.ndarray) or not isinstance(att_actions, np.ndarray):
            raise ValueError("Actions are not in numpy array.")

        # Define the rewards.
        reward_def = 0
        reward_att = 0

        for layer in range(self.num_layers):
            print("----")
            # deployment
            num_open_nodes = int(self.open_nodes[layer] * self.num_nodes_per_layer[layer])
            print("num_open_nodes:", num_open_nodes)
            # Sample a set of nodes for each player and let defender's action override attacker's action.
            att_deploy = set(random.sample(list(range(num_open_nodes)), int(num_open_nodes * att_actions[self.num_layers + layer])))
            def_deploy = set(random.sample(list(range(num_open_nodes)), int(num_open_nodes * def_actions[self.num_layers + layer])))
            print("att_deploy:", att_deploy)
            print("def_deploy:", def_deploy)

            att_deploy_rate = len(att_deploy.difference(def_deploy)) / self.num_nodes_per_layer[layer]
            def_deploy_rate = len(def_deploy) / self.num_nodes_per_layer[layer]

            print("att_deploy_rate", att_deploy_rate)
            print("def_deploy_rate", def_deploy_rate)

            # attack
            num_normal_nodes = int(self.normal_nodes[layer] * self.num_nodes_per_layer[layer])
            num_active_nodes = int((self.normal_nodes[layer] + self.compromised_nodes[layer]) * self.num_nodes_per_layer[layer])
            num_att_deploy_nodes = int(self.att_deploy[layer] * self.num_nodes_per_layer[layer])

            print("3:", num_normal_nodes, num_active_nodes, num_att_deploy_nodes)

            att_compromised = set(
                random.sample(list(range(num_active_nodes)), int(num_normal_nodes * att_actions[layer] * self.active_rate[layer] + self.num_nodes_per_layer[layer] * self.compromised_nodes[layer])))
            def_exclude = set(
                random.sample(list(range(num_active_nodes)), int(num_active_nodes * def_actions[layer])))
            att_deploy = set(random.choices(list(att_compromised), k=num_att_deploy_nodes))

            print("att_compromised:", att_compromised)
            print("def_exclude:", def_exclude)
            print("att_deploy:", att_deploy)

            att_compromised_rate = len(att_compromised.difference(def_exclude)) / self.num_nodes_per_layer[layer]
            def_exclude_rate = len(def_exclude) / self.num_nodes_per_layer[layer]
            att_deploy_old_rate = len(att_deploy.difference(def_exclude)) / self.num_nodes_per_layer[layer]

            print("att_compromised_rate:", att_compromised_rate)
            print("def_exclude_rate:", def_exclude_rate)
            print("att_deploy_old_rate:", att_deploy_old_rate)

            new_compromised_rate = att_compromised_rate + att_deploy_rate
            new_open_rate = self.open_nodes[layer] - att_deploy_rate - def_deploy_rate + def_exclude_rate
            assert new_open_rate + new_compromised_rate <= 1
            new_normal_rate = 1 - new_open_rate - new_compromised_rate

            # Update layer information
            self.compromised_nodes[layer] = new_compromised_rate
            self.normal_nodes[layer] = new_normal_rate
            self.open_nodes[layer] = new_open_rate

            # Attacker's payoff
            reward_att += self.a_attack_cost[layer] * att_actions[layer] \
                          + self.a_deploy_cost[layer] * att_actions[self.num_layers + layer] \
                          + self.a_maintain_cost[layer] * self.att_deploy[layer]
            self.att_deploy[layer] = att_deploy_old_rate + att_deploy_rate

            # Defender's payoff
            reward_def += self.d_exclude_cost[layer] * def_actions[layer] \
                          + self.d_deploy_cost[layer] * def_actions[self.num_layers + layer] \
                          + self.d_maintain_cost[layer] * self.def_deploy[layer]
            self.def_deploy[layer] = new_normal_rate + new_compromised_rate - self.att_deploy[layer]

            print("REW att:", self.a_attack_cost[layer] * att_actions[layer],
                  self.a_deploy_cost[layer] * att_actions[self.num_layers + layer],
                  self.a_maintain_cost[layer] * self.att_deploy[layer])
            print("REW def:", self.d_exclude_cost[layer] * def_actions[layer],
                  self.d_deploy_cost[layer] * def_actions[self.num_layers + layer],
                  self.d_maintain_cost[layer] * self.def_deploy[layer])

        reward_att += self.a_alpha * np.prod(self.compromised_nodes)
        reward_def -= self.d_beta * np.prod(self.compromised_nodes)

        print("REW att2:", self.a_alpha * np.prod(self.compromised_nodes))
        print("REW def2:", self.d_beta * np.prod(self.compromised_nodes))

        if np.prod(self.normal_nodes + self.compromised_nodes) < self.usage_threshold:
            print("PENALTY")
            reward_def += self.d_penalty

        # Update observations
        self.update_graph_state()

        return self.get_def_observation(), self.get_att_observation(), reward_def, reward_att, self.get_graph_state()


    def update_graph_state(self):
        """
        Get the state and players' obervations.
        state is a list [ratio of normal nodes * num_layers ratio of compromised nodes * num_layers]
        """
        self.state = np.concatenate((self.normal_nodes, self.compromised_nodes), axis=None)
        self.def_obs_normal = []
        self.def_obs_compromised = []
        for layer in range(self.num_layers):
            # False alarm
            false_alarms = np.random.binomial(self.num_nodes_per_layer[layer] * self.normal_nodes[layer], self.false_alarm[layer]) / self.num_nodes_per_layer[layer]
            # False negative
            false_negative = np.random.binomial(self.num_nodes_per_layer[layer] * self.compromised_nodes[layer], self.false_negative[layer]) / self.num_nodes_per_layer[layer]
            self.def_obs_normal.append(self.normal_nodes[layer] - false_alarms + false_negative)
            self.def_obs_compromised.append(self.compromised_nodes[layer] + false_alarms - false_negative)

        self.def_obs = np.concatenate((self.def_obs_normal, self.def_obs_compromised), axis=None)

    def get_graph_state(self):
        return self.state

    def get_def_observation(self):
        return self.def_obs

    def get_att_observation(self):
        return self.state




















