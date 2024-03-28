from tianshou.data import Collector, VectorReplayBuffer

class MAPolicyTrainer_v2():
    """
    MAPolicyTrainer_v2 differs from v1 in costomized training without using existing Trainers
    (OnPolicyTrainer and OffPolicyTrainer)
    """
    def __init__(self,
                 env,
                 on_policy,
                 best_response_kwargs,
                 train_envs,
                 test_envs,
                 test_in_train=False):
        self.env = env # self.env is a tianshou PettingZoo wrapper. env.env is a PttingZoo env.

        self.init_collector = False
        self.init_trainer = False

        # Whether the oracle is an on-policy algorithm.
        # This is a placeholder to be compatible with v1.
        self.on_policy = on_policy

        self.train_envs = train_envs
        self.test_envs = test_envs
        self.test_in_train = test_in_train

        # Training parameters
        # TODO: make this non-hard-coding.
        self.best_response_kwargs = best_response_kwargs
        self.iter_counter = 0 # Num of train steps
        self.step_counter = 0 # Num of steps an agent takes in the env.

        self.epsilon_start = self.best_response_kwargs["epsilon_start"] # Initial exploration rate.
        self.epsilon_end = self.best_response_kwargs["epsilon_end"] # End exploration rate.
        self.epsilon_decay_duration = int(self.best_response_kwargs["epsilon_decay_duration"])

    def get_epsilon(self, is_evaluation, power=1.0):
        """Returns the evaluation or decayed epsilon value."""
        if is_evaluation:
            return 0.0
        decay_steps = min(self.iter_counter, self.epsilon_decay_duration)
        decayed_epsilon = (
                self.epsilon_end + (self.epsilon_start - self.epsilon_end) *
                (1 - decay_steps / self.epsilon_decay_duration) ** power)
        return decayed_epsilon


    def setup_collector(self, master_policy):
        # Environment setup
        #TODO: Check if on-policy algorithm needs the replay buffer. Specific to DQN.
        train_collector = Collector(
            master_policy,
            self.train_envs,
            VectorReplayBuffer(self.best_response_kwargs["replay_buffer_size"], len(self.train_envs)),
            exploration_noise=True,
        )

        test_collector = Collector(master_policy,
                                   self.test_envs,
                                   exploration_noise=True)


        # # TODO: Do we need this? No, this will collect data from random policies.
        # train_collector.collect(n_step=self.best_response_kwargs["batch_size"] * self.best_response_kwargs["num_parallel_envs"])  # batch size * training_num

        return train_collector, test_collector

    def update_policy_in_collector(self, new_master_policy):
        self.train_collector.policy = new_master_policy
        if self.test_collector is not None:
            self.test_collector.policy = new_master_policy

    def reset(self):
        """
        Reset the mapolicy_trainer.
        """
        self.iter_counter = 0  # Num of train steps
        self.step_counter = 0  # Num of steps an agent takes in the env.
        self.reset_collector()

    def reset_collector(self):
        """
        Reset the collectors. Reset collect will reset train_env and test_env implicitly.
        """
        if self.init_collector:
            self.train_collector.reset()
            self.test_collector.reset()

    def train_one_step(self, master_policy, n_episode=1, test_freq=100, updater_per_episode=True):
        # One step of training:
        # (1) Collect n episodes;
        # (2) Test the current policy if test_in_train;
        # (3) Update a policy and return loss.

        master_policy.set_eps(self.get_epsilon(is_evaluation=False))
        collect_result = self.train_collector.collect(n_episode=n_episode)
        # collect_result = {
        #     "n/ep": episode_count,
        #     "n/st": step_count,
        #     "rews": rews,
        #     "lens": lens,
        #     "idxs": idxs,
        #     "rew": rew_mean,
        #     "len": len_mean,
        #     "rew_std": rew_std,
        #     "len_std": len_std,
        # } Tianshou 0.5.1
        self.step_counter += collect_result["n/st"]

        if self.step_counter <= self.best_response_kwargs["min_buffer_size_to_learn"]:
            return "C", None, None


        if self.test_in_train and self.iter_counter % test_freq == 0:
            master_policy.set_eps(self.get_epsilon(is_evaluation=True))
            test_result = self.test_collector.collect(n_episode=100)
            # back to training eps
            master_policy.set_eps(self.get_epsilon(is_evaluation=False))
        else:
            test_result = None

        # Update a policy for every collection or learn_every steps.
        if updater_per_episode:
            losses = master_policy.update(self.best_response_kwargs["batch_size"],
                                          self.train_collector.buffer)
        else:
            if self.step_counter % self.best_response_kwargs["learn_every"] == 0:
                losses = master_policy.update(self.best_response_kwargs["batch_size"],
                                              self.train_collector.buffer)
            else:
                losses = "Not update"
        self.iter_counter += 1

        return losses, collect_result, test_result


    def run(self, master_policy):
        """
        Running the trainer for one step.
        """
        if not self.init_collector:
            self.train_collector, self.test_collector = self.setup_collector(master_policy)
            self.init_collector = True
        else:
            self.update_policy_in_collector(new_master_policy=master_policy)

        losses, train_result, test_result = self.train_one_step(master_policy)
        if train_result is None:
            return losses, train_result, None, None

        train_rew = train_result['rews'].mean(axis=0)[master_policy.learning_players_id]
        if test_result is not None:
            test_rew = test_result['rews'].mean(axis=0)[master_policy.learning_players_id]
        else:
            test_rew = None

        return losses, train_result, train_rew, test_rew


