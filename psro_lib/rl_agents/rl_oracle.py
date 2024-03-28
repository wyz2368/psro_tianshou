import copy
import torch
from tianshou.utils.net.common import Net

from psro_lib.rl_agents.master_policy import MultiAgentPolicyManager_PSRO

def print_params(policy):
    """
    For test purpose only.
    """
    nn = policy.model
    for p in nn.parameters():
        print(p.data)
        break

# Freeze the weights of policies. If a freezed policy is being trained,
# then there will be an error saying no grad_fn. This is used for fixing
# other players' strategies.
def freeze_all(policies):
  """
  Freezes all policies within policy_per_player.
  """
  for policy in policies:
      policy.train(mode=False)
      freeze_module(policy)

def freeze_module(policy):
    """
    freeze nn.module in Pytorch.
    """
    nn = policy.model
    for p in nn.parameters():
        p.requires_grad = False

def unfreeze_all(policies):
    """
      Unfreezes all policies within policy_per_player.
    """
    for policy in policies:
        policy.train(mode=True)
        unfreeze_module(policy)

def unfreeze_module(policy):
    """
    freeze nn.module in Pytorch.
    """
    nn = policy.model
    for p in nn.parameters():
        p.requires_grad = True

def set_eps_zeros(policies):
    """
    Set the exploration rate for a list of policies.
    """
    for policy in policies:
        policy.set_eps(0.0)


class RLOracle:
    """Oracle handling Approximate Best Responses computation."""

    def __init__(self,
                 env,
                 trainer, # An MApolicyTrainer object in mapolicy_trainer_v2.py
                 best_response_class, # DRL policy.
                 best_response_kwargs, # Hyparams for DRL policy.
                 number_training_episodes,
                 sigma=0.0, # Noise for copying strategies.
                 verbose=True,
                 **kwargs):

        self.env = env
        self.num_players = len(env.agents)

        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs

        self.mapolicy_trainer = trainer
        self._number_training_episodes = int(number_training_episodes)

        self.sigma = sigma # Noise for copying strategies.
        self.init_master_policies = False

        # A flag in case any train/test info should be recorded.
        self.verbose = verbose


    def generate_new_policies(self,
                              copy_from_prev=False,
                              old_policies=None,
                              copy_with_noise=False):
        # Create a container of new policies.
        new_policies = []

        # Two ways to create a new policy: copy from the policy from previous iteration or create a new one.
        for player in range(len(self.env.agents)):
            # Not copy from previous iterations.
            if not copy_from_prev:
                # Initialize a net.
                net = Net(state_shape=self._best_response_kwargs["state_size"],
                          action_shape=self._best_response_kwargs["num_actions"],
                          hidden_sizes=[self._best_response_kwargs["hidden_layers_sizes"]] * self._best_response_kwargs["hidden_layers"],
                          device="cuda" if torch.cuda.is_available() else "cpu").to("cuda" if torch.cuda.is_available() else "cpu")

                # Initialize an optimizer.
                if self._best_response_kwargs["optimizer_str"] == "adam":
                    optim = torch.optim.Adam(net.parameters(), lr=self._best_response_kwargs["learning_rate"])
                elif self._best_response_kwargs["optimizer_str"] == "sgd":
                    optim = torch.optim.SGD(net.parameters(), lr=self._best_response_kwargs["learning_rate"])
                else:
                    raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

                # Create a DRL policy.
                # TODO: target_update_freq may only work for DQN.
                policy = self._best_response_class(model=net,
                                                   optim=optim,
                                                   discount_factor=self._best_response_kwargs["discount_factor"],
                                                   estimation_step=self._best_response_kwargs["estimation_step"], # The number of steps to look ahead.
                                                   target_update_freq=self._best_response_kwargs["update_target_network_every"])

                policy.set_agent_id(self.env.agents[player])

            # Copy from previous iterations.
            else:
                if old_policies is None:
                    raise ValueError("No previous policy can be duplicated.")
                target_policy = old_policies[player][-1]
                if not isinstance(target_policy, self._best_response_class):
                    raise ValueError("The target policy does not belong to the best response class.")
                policy = copy.deepcopy(target_policy) # Copy the policy from last iteration.
                if copy_with_noise:
                    with torch.no_grad():
                        for param in policy.model.parameters():
                            param.add_(torch.randn(param.size()) * self.sigma)

                unfreeze_module(policy)

            new_policies.append(policy)

        return new_policies


    def train(self,
              old_policies,
              meta_probabilies,
              strategy_sampler):
        """
        Training new policies by best responses to other players' policies sampled according to meta_probabilities.
        :param old_policies: a list of lists of policies from previous PSRO iterations.
        :param meta_probabilies: a list of mixed strategies, one for each player.
        :param strategy_sampler: a function sampling strategies probabilistically.
        """
        data = {}
        for learning_player in range(self.num_players):
            data[learning_player] = {}
            data[learning_player]["losses"] = []
            data[learning_player]["train_rew"] = []
            data[learning_player]["test_rew"] = []

        for i in range(self._number_training_episodes):
            # Sample a strategy profile. A list of policies.
            sampled_policies = strategy_sampler(old_policies, meta_probabilies)

            #TODO: make the train env and test env share among players.
            for learning_player in range(self.num_players):
                # Construct a new Master policy based on the sampled policies. This is achieved by replacing non-learning-player's policies.
                self.master_policies[learning_player].update_policies_except_the_learning_players(sampled_policies)
                # Train the master policy for a few iterations.
                losses, train_result, train_rew, test_rew = self.mapolicy_trainer.run(self.master_policies[learning_player])
                if losses != "C":
                    data[learning_player]["losses"].append(losses.values())
                    data[learning_player]["train_rew"].append(train_rew)
                    data[learning_player]["test_rew"].append(test_rew)

        if self.verbose:
            for learning_player in range(self.num_players):
                print("-----------------")
                print("Losses1:", data[learning_player]["losses"][:50])
                print("Losses2:", data[learning_player]["losses"][-50:])
                print("train_rew:", data[learning_player]["train_rew"][-50:])
                # print("test_rew:", data[learning_player]["test_rew"])


    def __call__(self,
                 env,
                 old_policies,
                 meta_probabilities,
                 strategy_sampler,
                 copy_from_prev,
                 *args,
                 **kwargs):

        # Generate a list of initial policies, one for each player.
        new_policies = self.generate_new_policies(copy_from_prev=copy_from_prev,
                                                  old_policies=old_policies)

        # print_params(new_policies[0])
        # print_params(new_policies[1])
        # print("------")

        # print_params(new_policies[1])

        # Generate a list of initial master policies if they haven't been created, one for each player.
        if not self.init_master_policies:
            # Create a list of master policies, one for each player.
            self.master_policies = []
            for learning_player in range(self.num_players):
                self.master_policies.append(MultiAgentPolicyManager_PSRO(policies=new_policies,
                                                                         env=env,
                                                                         learning_players_id=learning_player))
            self.init_master_policies = True
        else:
            # Update the polices in master policies.
            for learning_player in range(self.num_players):
                self.master_policies[learning_player].update_policies(new_policies=new_policies)


        # Train new policies.
        self.mapolicy_trainer.reset()
        self.train(strategy_sampler=strategy_sampler,
                   old_policies=old_policies,
                   meta_probabilies=meta_probabilities)

        # Freeze the new policies to keep their weights static.
        #TODO: Chekc if we need freeze_all, since MAPOLICY only update the learning strategy.
        freeze_all(new_policies)
        # Set all policies with eps = 0.0 .
        set_eps_zeros(new_policies)
        trained_policies = [[pol] for pol in new_policies] # A list of lists of new policies (One list per player)

        # print_params(new_policies[0])
        # print_params(trained_policies[0][0])
        # print_params(new_policies[1])
        # print_params(trained_policies[1][0])
        # print(trained_policies[0][0] is trained_policies[1][0])

        # print_params(new_policies[1])

        return trained_policies







