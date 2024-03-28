import numpy as np
import random
from applications.mixnet.graph_generator import Graph


def verify_graph():
    random.seed(0)
    np.random.seed(0)

    graph = Graph(link_probability=0.5,
                  nodes_per_layer=[2, 2])
    print("Nums of Nodes:", graph.get_num_nodes())
    a_deploy = 0
    a_attack = 0
    d_deploy = 0
    d_main = 0

    for i in range(4):
        node = graph.id_to_node[i]
        a_deploy += node.a_deploy_cost
        a_attack += node.a_attack_cost
        d_deploy += node.d_deploy_cost
        d_main += node.d_maintain_cost

    print("a_deploy:", a_deploy)
    print("a_attack:", a_attack)
    print("d_deploy:", d_deploy)
    print("d_deploy:", d_main)

    for i in range(4):
        print("==========var==========")
        node = graph.get_node_by_id(i)
        print(vars(node))

    print("adj:", graph.adjacency_matrix)
    print("orig state:", graph.get_graph_state())
    print("===========Test Actions============")
    print("___deploy___")
    # def_actions = np.array([1,1,1,1,1,1,1,1,1])
    # att_actions = np.array([1,1,1,1,1,1,1,1,1])
    def_actions = np.array([1, 1, 1, 1])
    att_actions = np.array([1, 1, 1, 1])
    obs_def, obs_att, reward_def, reward_att, state = graph.step(def_actions, att_actions)
    print("state:", state)
    print("def obs:", obs_def)
    print("att obs:", obs_att)
    print("def rew:", reward_def)
    print("att rew:", reward_att)

    # print("___exclude___")
    # def_actions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    # att_actions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    # graph.step(def_actions, att_actions)
    # print("state:", graph.get_graph_state())
    # print("def obs:", graph.get_def_observation())
    # print("att obs:", graph.get_att_observation())

    print("___attack___")
    print("orig state:", graph.get_graph_state())
    # def_actions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    # att_actions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    def_actions = np.array([0, 0, 0, 0])
    att_actions = np.array([1, 1, 1, 1])
    obs_def, obs_att, reward_def, reward_att, state = graph.step(def_actions, att_actions)
    print("state:", state)
    print("def obs:", obs_def)
    print("att obs:", obs_att)
    print("def rew:", reward_def)
    print("att rew:", reward_att)

    print("___attack2___")
    print("orig state:", graph.get_graph_state())
    # def_actions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    # att_actions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    def_actions = np.array([0, 0, 0, 0])
    att_actions = np.array([1, 1, 1, 1])
    obs_def, obs_att, reward_def, reward_att, state = graph.step(def_actions, att_actions)
    print("state:", state)
    print("def obs:", obs_def)
    print("att obs:", obs_att)
    print("def rew:", reward_def)
    print("att rew:", reward_att)


if __name__ == "__main__":
    verify_graph()
