from tianshou.policy import MultiAgentPolicyManager

#TODO: change to agents
class MultiAgentPolicyManager_PSRO(MultiAgentPolicyManager):
    """
    This class inherits the MultiAgentPolicyManager class in Tianshou, with functionalities:
    1) Only update one player's strategy.
    2) Update the policies.

    """
    def __init__(self,
                 policies,
                 env,
                 learning_players_id,
                 **kwargs):
        super(MultiAgentPolicyManager_PSRO, self).__init__(policies, env, **kwargs)

        self.env = env

        # Learning player ID. Index from 0. ID 0 corresponds to player 1.
        self.learning_players_id = learning_players_id
        self.learning_player_string = self.env.agents[learning_players_id]

    def set_eps(self, eps):
        self.policies[self.learning_player_string].set_eps(eps)

    # TODO: how to combine sampling strategies with this class?
    def update_learning_players(self, new_learning_players_id):
        self.learning_players_id = new_learning_players_id
        self.learning_player_string = self.env.agents[new_learning_players_id]

    def update_policies(self, new_policies):
        self.policies = dict(zip(self.env.agents, new_policies, strict=True))

    def update_policies_except_the_learning_players(self, sampled_policies): #TODO: check the consistency of dict policies and Spiel policies
        """
        Mimicking that Openspiel first samples a strategy profile and then replaces the learning player's policy with the new policy.
        """
        updated_policies = []
        for player, player_str in enumerate(self.env.agents):
            if player == self.learning_players_id:
                # Learning player's policy remains unchanged.
                updated_policies.append(self.policies[player_str])
            else:
                # Only update non-learning-player policies.
                updated_policies.append(sampled_policies[player])

        self.update_policies(updated_policies)

    def get_single_policy(self, player_id):
        """
        Return player_id's policy.
        """
        return self.policies[self.env.agents[player_id]]

    def get_all_policies(self):
        """
        Order the policies in terms of players' ids and return.
        """
        policies = []
        for player, player_str in enumerate(self.env.agents):
            policies.append(self.policies[player_str])
        return policies

    def learn(self, batch, *args, **kwargs): #TODO: See if we can set master policy training 1 episode per time and updates periodically.

        """
        Only update the learning player's policy.
        """
        results = {}
        # agent_id = self.env.agents[self.learning_players_id]
        data = batch[self.learning_player_string]
        policy = self.policies[self.learning_player_string]
        if not data.is_empty():
            out = policy.learn(batch=data, **kwargs)
            for k, v in out.items():
                results[self.learning_player_string + "/" + k] = v
        return results #TODO: check if we need to return empty {} for non-training agents.

    # Overload the train function, such that only the learning player's train flag open. #TODO: Do we need this?
    # def train(self, mode=True):
    #     """Set each internal policy in training mode."""
    #     for policy in self.policies.values():
    #         policy.train(mode)
    #     return self


