from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer

from psro_lib.rl_agents.utils import get_env_factory
from psro_lib.rl_agents.customized_trainers import OffpolicyTrainer_customized, OnpolicyTrainer_customized



class MApolicyTrainer():
    def __init__(self, env, on_policy, num_parallel_envs=5):
        self.env = env
        self.num_parallel_envs = num_parallel_envs

        self.init_collector = False
        self.init_trainer = False

        #TODO: Check how trainer calls.
        if on_policy:
            self.trainer_class = OnpolicyTrainer_customized
        else:
            self.trainer_class = OffpolicyTrainer_customized

    def create_trainer(self,
                        master_policy,
                        train_collector,
                        test_collector=None,
                        stop_fn=None,
                        train_fn=None,
                        test_fn=None,
                        save_best_fn=None,
                        reward_metric=None):


        #TODO: See if we can set master policy training 1 episode per time and updates periodically.

        #TODO: How to align the hyerparameters with Openspiel?
        #TODO: Some of the arguments are not required in the lastest version 1.00.
        self.trainer = self.trainer_class(policy=master_policy,
                                        train_collector=train_collector,
                                        test_collector=test_collector,
                                        max_epoch=40, # The maximum number of epochs for training.
                                        step_per_epoch=1000, # The number of transitions collected per epoch.
                                        step_per_collect=10, # The number of transitions the collector would collect before the network update.
                                        episode_per_test=100, # The number of episodes for one policy evaluation.
                                        batch_size=32,
                                        train_fn=train_fn,
                                        test_fn=test_fn,
                                        stop_fn=stop_fn,
                                        save_best_fn=save_best_fn,
                                        update_per_step=1.0,
                                        test_in_train=False,
                                        reward_metric=reward_metric)



    def setup_training(self, master_policy, num_parallel_envs=5, test_in_train=False):
        # Environment setup
        train_envs = DummyVectorEnv([get_env_factory(self.env.metadata["name"]) for _ in range(num_parallel_envs)])

        train_collector = Collector(
            master_policy,
            train_envs,
            VectorReplayBuffer(20000, len(train_envs)),
            exploration_noise=True,
        )
        if test_in_train:
            test_envs = DummyVectorEnv([get_env_factory(self.env.metadata["name"]) for _ in range(num_parallel_envs)])
            test_collector = Collector(master_policy, test_envs, exploration_noise=True)
        else:
            test_collector = None
        # policy.set_eps(1)
        train_collector.collect(n_step=32 * num_parallel_envs)  # batch size * training_num #TODO: Do we need this?

        return train_collector, test_collector

    def update_policy_in_collector(self, new_master_policy):
        self.train_collector.policy = new_master_policy
        if self.test_collector is not None:
            self.test_collector.policy = new_master_policy

    def update_policy_in_trainer(self, new_master_policy):
        self.trainer.policy = new_master_policy


    #TODO: Is this really slow? This will result in an issue: Replace the policy of collector everytime
    #TODO: Does the policy in data structure in collector matter?
    def run(self, master_policy):
        if not self.init_collector:
            self.train_collector, self.test_collector = self.setup_training(master_policy)
            self.init_collector = True
        else:
            self.update_policy_in_collector(new_master_policy=master_policy)

        if not self.init_trainer:
            self.create_trainer(master_policy=master_policy,
                                 train_collector=self.train_collector,
                                 test_collector=self.test_collector)

            self.init_trainer = True

        else:
            self.update_policy_in_trainer(master_policy)

        self.trainer.run()

