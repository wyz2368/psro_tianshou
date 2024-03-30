import numpy as np
import random

from applications.mixnet_homo.graph_generator import HOMOGraph

def run_graph():
    graph = HOMOGraph(
        num_layers=3,
        num_nodes_per_layer=[1000, 1000, 1000],
        false_alarm=[0.3, 0.2, 0.3], # false alarm rate
        false_negative=[0.2, 0.2, 0.2], # false negative rate
        a_attack_cost=[-400, -500, -400], # attacker's attacking cost
        a_deploy_cost=[-200, -200, -200], # attacker's deployment cost
        a_maintain_cost=[-50, -50, -50], # attacker's maintaining cost
        active_rate=[0.3, 0.2, 0.3], # the possibility of sucessfully activate a node
        d_exclude_cost=[-200, -200, -200], # defender's cost on excluding a node
        d_deploy_cost=[-100, -100, -100], # defender's deployment cost
        d_maintain_cost=[-30, -30, -30],
        usage_threshold=0.03, # the lower bound of usage without penalty
        d_penalty=-50, # defender's penalty for insufficient usage
        a_alpha=50000,  # coefficient for the reward
        d_beta=50000,
        seed=100)

    print("Normal:", graph.normal_nodes)
    print("Compromised:", graph.compromised_nodes)
    print("Open:", graph.open_nodes)

    # print("====== Attack =======")
    # def_actions = [0] * 6
    # att_actions = [0.5, 0.5, 0.5, 0, 0, 0]
    #
    # obs_def, obs_att, reward_def, reward_att, state = graph.step(np.array(def_actions), np.array(att_actions))
    # print("-----")
    # print("state:", state)
    # print("def obs:", obs_def)
    # print("att obs:", obs_att)
    # print("def rew:", reward_def)
    # print("att rew:", reward_att)

    print("====== Attack + deploy =======")
    def_actions = [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    att_actions = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]

    obs_def, obs_att, reward_def, reward_att, state = graph.step(np.array(def_actions), np.array(att_actions))
    print("-----")
    print("state:", state)
    print("def obs:", obs_def)
    print("att obs:", obs_att)
    print("def rew:", reward_def)
    print("att rew:", reward_att)


if __name__ == "__main__":
    run_graph()


