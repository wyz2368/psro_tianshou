import numpy as np
import random

def check_probability_sum2one(normal_nodes, compromised_nodes, open_nodes):
    pass


class HOMOGraph():
    def __init__(self,
                 num_layers,
                 num_nodes_per_layer,
                 false_alarm,
                 false_negative,
                 a_attack_cost,
                 a_deploy_cost,
                 a_maintain_cost,
                 a_alpha,
                 active_rate,
                 d_exclude_cost,
                 d_deploy_cost,
                 d_maintain_cost,
                 usage_threshold,
                 d_penalty,
                 d_beta,
                 normal_nodes=None,
                 compromised_nodes=None):

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
        if normal_nodes is not None:
            self.normal_nodes = normal_nodes
        else:
            self.normal_nodes = np.random.uniform(0, 1, num_layers)

        if compromised_nodes is not None:
            self.compromised_nodes = compromised_nodes
        else:
            self.compromised_nodes = np.zeros(num_layers)
            for i in range(num_layers):
                self.compromised_nodes[i] = np.random.uniform(0, self.normal_nodes[i])

        self.open_nodes = 1 - self.normal_nodes - self.compromised_nodes

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
            # deployment
            num_open_nodes = int(self.open_nodes[layer] * self.num_nodes_per_layer[layer])
            att_deploy = set(random.sample(range(num_open_nodes), num_open_nodes * att_actions[self.num_layers + layer]))
            def_deploy = set(random.sample(range(num_open_nodes), num_open_nodes * def_actions[self.num_layers + layer]))

            att_deploy_rate = len(att_deploy.difference(def_deploy)) / self.num_nodes_per_layer[layer]
            def_deploy_rate = len(def_deploy) / self.num_nodes_per_layer[layer]

            # attack
            num_normal_nodes = int(self.normal_nodes[layer] * self.num_nodes_per_layer[layer])
            num_active_nodes = int((self.normal_nodes[layer] + self.compromised_nodes[layer]) * self.num_nodes_per_layer[layer])
            att_compromised = set(
                random.sample(range(num_active_nodes), num_normal_nodes * att_actions[layer] * self.active_rate[layer] + self.num_nodes_per_layer[layer] * self.compromised_nodes[layer]))
            def_exclude = set(
                random.sample(range(num_active_nodes), num_active_nodes * def_actions[layer]))

            att_compromised_rate = len(att_compromised.difference(def_exclude)) / self.num_nodes_per_layer[layer]
            def_exclude_rate = len(def_exclude) / self.num_nodes_per_layer[layer]

            new_compromised_rate = att_compromised_rate + att_deploy_rate
            new_open_rate = self.open_nodes[layer] - att_deploy_rate - def_deploy_rate + def_exclude_rate
            new_normal_rate = 1 - new_open_rate - new_compromised_rate

            self.compromised_nodes[layer] = new_compromised_rate
            self.normal_nodes[layer] = new_normal_rate
            self.open_nodes[layer] = new_open_rate

            reward_att += self.a_attack_cost[layer] * att_actions[layer] \
                          + self.a_deploy_cost * att_actions[self.num_layers + layer] \
                          + self.a_maintain_cost * self.att_deploy[layer]
            self.att_deploy[layer] = att_deploy_rate
            reward_def += self.d_exclude_cost[layer] * def_actions[layer] \
                          + self.d_deploy_cost * def_actions[self.num_layers + layer] \
                          + self.d_maintain_cost * self.def_deploy[layer]
            self.def_deploy[layer] = new_normal_rate + new_compromised_rate - self.att_deploy[layer]

        reward_att += self.a_alpha * np.prod(self.compromised_nodes)
        reward_def += -self.d_beta * np.prod(self.compromised_nodes)

        if np.prod(self.normal_nodes + self.compromised_nodes) < self.usage_threshold:
            reward_def += self.d_penalty

    def update_graph_state(self):
        """
        Get the state and players' obervations.
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




















