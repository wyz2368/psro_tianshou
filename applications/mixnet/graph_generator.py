import numpy as np
import random
from applications.mixnet.utils import load_pkl, save_pkl, compute_linkages

# Attacker's rewards/costs
A_DEPLOY_COST_MIN = -2.0  # the cost of deploying a new server for the attacker
A_DEPLOY_COST_MAX = -5.0
A_ATTACK_COST_MIN = -2.0  # the cost of attacking a node for the attacker
A_ATTACK_COST_MAX = -10.0
A_MAINTAIN_COST_MIN = -1.0  # the cost of maintaining a server for the attacker
A_MAINTAIN_COST_MAX = -2.0
A_LINK_REW_MIN = 10.0  # the reward of linking users for the attacker
A_LINK_REW_MAX = 20.0
# Defender's rewards/costs
D_DEPLOY_COST_MIN = -2.0  # the cost of deploying a new server for the defender
D_DEPLOY_COST_MAX = -5.0
D_MAINTAIN_COST_MIN = -1.0  # the cost of maintaining a server for the defender
D_MAINTAIN_COST_MAX = -2.0
D_LINK_PENALTY_MIN = -10.0  # the penalty of having users linked for the defender
D_LINK_PENALTY_MAX = -20.0
D_USAGE_PENALTY_MIN = -20.0  # the penalty of not having enough paths for the defender
D_USAGE_PENALTY_MAX = -30.0
D_DEFEND_REW_MIN = 3.0  # the reward of successfully excluding a compromised node
D_DEFEND_REW_MAX = 6.0
# Noisy observations/attack success rate
FALSE_NEGATIVE_MIN = 0.1  # prob of sending positive signal if node is active
FALSE_NEGATIVE_MAX = 0.4
FALSE_ALARM_MIN = 0.1  # prob of sending positive signal if node is inactive(false alarm)
FALSE_ALARM_MAX = 0.4
ACTPROB_MIN = 0.1
ACTPROB_MAX = 0.9



class Node():
    """
    This node class represents the node in the mixnet.
    """
    def __init__(self,
                 id, # id of a node in the mixnet.
                 state=-1, # 0: controlled by the defender 1: controlled by the attacker -1:not deployed.
                 control=-1,
                 # attacker's rewards/costs
                 a_deploy_cost=0.0, # the cost of deploying a new server for the attacker
                 a_attack_cost=0.0, # the cost of attacking a node for the attacker
                 a_maintain_cost=0.0, # the cost of maintaining a server for the attacker
                 a_link_rew=0.0, # the reward of linking users for the attacker
                 # defender's rewards/costs
                 d_deploy_cost=0.0,  # the cost of deploying a new server for the defender
                 d_maintain_cost=0.0, # the cost of maintaining a server for the defender
                 d_link_penalty=0.0, # the penalty of having users linked for the defender
                 d_usage_penalty=0.0, # the penalty of not having enough paths for the defender
                 d_defend_rew=0.0, # the reward of successfully excluding a compromised node
                 false_negative=1.0,  # prob of sending positive signal if node is active
                 false_alarm=0.0,  # prob of sending positive signal if node is inactive(false alarm)
                 actProb=1.0):  # prob of becoming corrupted if being attacked.

        # Properties of the node.
        self.id = id
        self.state = state
        # attacker's rewards/costs
        self.a_deploy_cost = a_deploy_cost  # the cost of deploying a new server for the attacker
        self.a_attack_cost = a_attack_cost  # the cost of attacking a node for the attacker
        self.a_maintain_cost = a_maintain_cost  # the cost of maintaining a server for the attacker
        self.a_link_rew = a_link_rew  # the reward of linking users for the attacker
        # defender's rewards/costs
        self.d_deploy_cost = d_deploy_cost  # the cost of deploying a new server for the defender
        self.d_maintain_cost = d_maintain_cost  # the cost of maintaining a server for the defender
        self.d_link_penalty = d_link_penalty  # the penalty of having users linked for the defender
        self.d_usage_penalty = d_usage_penalty  # the penalty of not having enough paths for the defender
        self.d_defend_rew = d_defend_rew  # the reward of successfully excluding a compromised node
        # Common probability distributions
        self.false_negative = false_negative
        self.false_alarm = false_alarm
        self.actProb = actProb

    # Getter methods
    @property
    def get_id(self):
        return self.id

    @property
    def get_state(self):
        return self.state

    @property
    def get_parents(self):
        return self.parents

    @property
    def get_children(self):
        return self.children

    @property
    def get_a_deploy_cost(self):
        return self.a_deploy_cost

    @property
    def get_a_attack_cost(self):
        return self.a_attack_cost

    @property
    def get_a_maintain_cost(self):
        return self.a_maintain_cost

    @property
    def get_a_link_rew(self):
        return self.a_link_rew

    @property
    def get_d_deploy_cost(self):
        return self.d_deploy_cost

    @property
    def get_d_maintain_cost(self):
        return self.d_maintain_cost

    @property
    def get_d_link_penalty(self):
        return self.d_link_penalty

    @property
    def get_d_usage_penalty(self):
        return self.d_usage_penalty

    @property
    def get_d_defend_rew(self):
        return self.d_defend_rew

    @property
    def get_false_negative(self):
        return self.false_negative

    @property
    def get_false_alarm(self):
        return self.false_alarm

    @property
    def get_actProb(self):
        return self.actProb

    # Setter methods
    def set_id(self, id):
        self.id = id

    def set_state(self, state):
        self.state = state

    def set_parents(self, parents):
        self.parents = parents

    def set_children(self, children):
        self.children = children

    def set_a_deploy_cost(self, cost):
        self.a_deploy_cost = cost

    def set_a_attack_cost(self, cost):
        self.a_attack_cost = cost

    def set_a_maintain_cost(self, cost):
        self.a_maintain_cost = cost

    def set_a_link_rew(self, reward):
        self.a_link_rew = reward

    def set_d_deploy_cost(self, cost):
        self.d_deploy_cost = cost

    def set_d_maintain_cost(self, cost):
        self.d_maintain_cost = cost

    def set_d_link_penalty(self, penalty):
        self.d_link_penalty = penalty

    def set_d_usage_penalty(self, penalty):
        self.d_usage_penalty = penalty

    def set_d_defend_rew(self, reward):
        self.d_defend_rew = reward

    def set_false_negative(self, prob):
        self.false_negative = prob

    def set_false_alarm(self, prob):
        self.false_alarm = prob

    def set_actProb(self, prob):
        self.actProb = prob


def uniform_sampling_params():
    state = random.choices([-1,0,1], weights = [6, 0, 0])[0]
    # Attacker's rewards/costs
    a_deploy_cost = random.uniform(A_DEPLOY_COST_MIN, A_DEPLOY_COST_MAX)
    a_attack_cost = random.uniform(A_ATTACK_COST_MIN, A_ATTACK_COST_MAX)
    a_maintain_cost = random.uniform(A_MAINTAIN_COST_MIN, A_MAINTAIN_COST_MAX)
    a_link_rew = random.uniform(A_LINK_REW_MIN, A_LINK_REW_MAX)
    # Defender's rewards/costs
    d_deploy_cost = random.uniform(D_DEPLOY_COST_MIN, D_DEPLOY_COST_MAX)
    d_maintain_cost = random.uniform(D_MAINTAIN_COST_MIN, D_MAINTAIN_COST_MAX)
    d_link_penalty = random.uniform(D_LINK_PENALTY_MIN, D_LINK_PENALTY_MAX)
    d_usage_penalty = random.uniform(D_USAGE_PENALTY_MIN, D_USAGE_PENALTY_MAX)
    d_defend_rew = random.uniform(D_DEFEND_REW_MIN, D_DEFEND_REW_MAX)
    # Noisy observations/attack success rate
    false_negative = random.uniform(FALSE_NEGATIVE_MIN, FALSE_NEGATIVE_MAX)
    false_alarm = random.uniform(FALSE_ALARM_MIN, FALSE_ALARM_MAX)
    actProb = random.uniform(ACTPROB_MIN, ACTPROB_MAX)

    return {
        'state': state,
        'a_deploy_cost': a_deploy_cost,
        'a_attack_cost': a_attack_cost,
        'a_maintain_cost': a_maintain_cost,
        'a_link_rew': a_link_rew,
        'd_deploy_cost': d_deploy_cost,
        'd_maintain_cost': d_maintain_cost,
        'd_link_penalty': d_link_penalty,
        'd_usage_penalty': d_usage_penalty,
        'd_defend_rew': d_defend_rew,
        'false_negative': false_negative,
        'false_alarm': false_alarm,
        'actProb': actProb
    }

def generate_nodes(nodes_per_layer, loaded_params=None):
    """
    Generate nodes with sampled params.
    """
    id_to_node = {}
    all_params = []
    for id in range(sum(nodes_per_layer)):
        if loaded_params is None:
            params = uniform_sampling_params()
        else:
            params = loaded_params[id]
        all_params.append(params)
        params["id"] = id
        node = Node(**params)
        id_to_node[id] = node

    return id_to_node, all_params

def generate_adjacency_matrix(layer_sizes, link_probability):
    """
    Generate edges for mixnets.
    :param layer_sizes: a list of numbers of nodes, one for each layer
    :param link_probability: The probability of having a link between two nodes.
    """
    num_layers = len(layer_sizes)
    total_nodes = sum(layer_sizes)

    # Initialize adjacency matrix as a dictionary
    adjacency_matrix = {i: [] for i in range(total_nodes)}

    # Connect nodes between adjacent layers
    for i in range(num_layers - 1):
        current_layer_size = layer_sizes[i]
        next_layer_size = layer_sizes[i + 1]
        current_index = sum(layer_sizes[:i])
        next_index = current_index + current_layer_size

        for j in range(current_index, next_index):
            # Connect each node to at least one node in the next layer
            random_neighbor = np.random.randint(next_index, next_index + next_layer_size)
            adjacency_matrix[j].append(random_neighbor)

            # Randomly connect additional nodes based on link_probability
            for k in range(next_index, next_index + next_layer_size):
                if k != random_neighbor and np.random.rand() < link_probability:
                    adjacency_matrix[j].append(k)

            adjacency_matrix[j] = list(sorted(adjacency_matrix[j]))

    return adjacency_matrix


class Graph():
    def __init__(self,
                 link_probability,
                 nodes_per_layer,
                 params_path=None,
                 adj_path=None):
        if params_path is not None:
            loaded_params = load_pkl(params_path)
            self.id_to_node, self.all_params = generate_nodes(nodes_per_layer=nodes_per_layer,
                                                                  loaded_params=loaded_params)
        else:
            self.id_to_node, self.all_params = generate_nodes(nodes_per_layer=nodes_per_layer)

        if adj_path is not None:
            self.adjacency_matrix = load_pkl(params_path)
        else:
            self.adjacency_matrix = generate_adjacency_matrix(layer_sizes=nodes_per_layer,
                                                              link_probability=link_probability)

        # These sets store node_id that they control.
        self.def_control = set()
        self.att_control = set()
        self.common_control = set()

        self.nodes_per_layer = nodes_per_layer
        self.min_paths = 1

        self.state = []
        for id in range(self.get_num_nodes()):
            node = self.id_to_node[id]
            state = node.state
            self.state.append(state)

        self.def_obs = []

    ############################## States ##############################

    def step(self, def_binary_actions, att_binary_actions):
        """
        Receive actions from the defender and the attacker, and update the state and reward.
        """
        if not isinstance(def_binary_actions, np.ndarray) or not isinstance(att_binary_actions, np.ndarray):
            raise ValueError("Actions are not in numpy array.")
        if len(def_binary_actions) != self.get_num_nodes() or len(att_binary_actions) != self.get_num_nodes():
            raise ValueError("Actions do not match the number of nodes.")
        if not np.array_equal(def_binary_actions, def_binary_actions.astype(bool)) or not np.array_equal(att_binary_actions, att_binary_actions.astype(bool)):
            raise ValueError("Actions are not binary.")

        # Define the rewards.
        reward_def = 0
        reward_att = 0

        # Obtain node ids, to which actions apply.
        def_actions = set(np.where(def_binary_actions == 1)[0])
        att_actions = set(np.where(att_binary_actions == 1)[0])
        valid_att_actions = att_actions - def_actions # def overrides att actions.

        for node_id in att_actions:
            reward_att += self.apply_att_action(node_id, valid_att_actions)

        for node_id in def_actions:
            reward_def += self.apply_def_action(node_id)

        # Maintaining costs after taking actions.
        for node_id in self.def_control:
            node = self.id_to_node[node_id]
            reward_def += node.d_maintain_cost

        for node_id in self.att_control:
            if node_id not in self.common_control:
                node = self.id_to_node[node_id]
                reward_att += node.a_maintain_cost

        # Success linkage of users.
        att_paths = compute_linkages(self.nodes_per_layer, self.att_control, self.adjacency_matrix)
        for path in att_paths:
            node_id = path[-1]
            node = self.id_to_node[node_id]
            reward_att += node.a_link_rew
            reward_def += node.d_link_penalty

        # Without usage penalty.
        if len(att_paths) == 0:
            all_paths = compute_linkages(self.nodes_per_layer, self.att_control.union(self.def_control), self.adjacency_matrix)
            if len(all_paths) <= self.min_paths: # Assume min_paths <= total_paths.
                reward_def += self.id_to_node[0].d_usage_penalty # Fix the penalty to be that of the node 0.

        # Update observations
        self.update_graph_state()

        return self.get_def_observation(), self.get_att_observation(), reward_def, reward_att, self.get_graph_state()


    def apply_def_action(self, node_id):
        """
        Apply defender's action.
        """
        node = self.id_to_node[node_id]
        state = node.state
        rew = 0.0
        if state == -1:
            node.set_state(0)
            self.def_control.add(node_id)
            rew += node.d_deploy_cost
        else:
            if node_id in self.att_control:
                rew += node.d_defend_rew
                self.att_control.remove(node_id)
            if node_id in self.def_control:
                self.def_control.remove(node_id)
            if node_id in self.common_control:
                self.common_control.remove(node_id)
            node.set_state(-1)

        return rew

    def apply_att_action(self, node_id, valid_att_actions):
        """
        Apply attacker's actions.
        """
        node = self.id_to_node[node_id]
        state = node.state
        rew = 0.0
        if state == -1:
            if node_id in valid_att_actions:
                node.set_state(1)
                self.att_control.add(node_id)
            rew += node.a_deploy_cost
        elif state == 0:
            rew += node.a_attack_cost
            if np.random.uniform(0, 1) <= node.actProb and node_id in valid_att_actions:
                node.set_state(1)
                self.att_control.add(node_id)
                self.common_control.add(node_id)
        else:
            # should add a hard constraint to disallow attacker to shut down defender's node.
            if node_id in self.att_control and node_id not in self.common_control and node_id in valid_att_actions: # Duo protection.
                node.set_state(-1)
                self.att_control.remove(node_id)

        return rew


    def update_graph_state(self):
        """
        Get the state and players' obervations.
        """
        self.state = []
        self.def_obs = []
        for id in range(self.get_num_nodes()):
            node = self.id_to_node[id]
            state = node.state
            self.state.append(state)

            if state == -1:
                self.def_obs.append(-1)
            elif state == 0:
                self.def_obs.append(1 if np.random.uniform(0, 1) <= node.false_alarm else 0)
            elif state == 1:
                self.def_obs.append(0 if np.random.uniform(0, 1) <= node.false_negative else 1)
            else:
                raise ValueError("Incorrect state.")


    ############################## General ##############################
    def get_num_nodes(self):
        return len(self.id_to_node)

    def get_node_by_id(self, id):
        return self.id_to_node[id]

    def get_graph_state(self):
        return self.state

    def get_def_observation(self):
        return self.def_obs

    def get_att_observation(self):
        return self.state

    def reset_state(self):
        for id in self.id_to_node:
            self.id_to_node[id].state = -1

    def save_graph(self, path):
        save_pkl(self.adjacency_matrix, path + "/adj.pkl")
        save_pkl(self.all_params, path + "/params.pkl")



















