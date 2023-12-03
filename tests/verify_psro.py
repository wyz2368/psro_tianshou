"""
This file verifies PSRO with random strategies.

There is no RL oracle, but adding random strategies each iteration.
"""
from psro_lib.meta_strategies import get_joint_strategy_from_marginals, general_get_joint_strategy_from_marginals
import numpy as np
from solution_solvers.nash_solver import lp_solver

#TODO: test uniform

ne = [np.array([1., 0.]), np.array([0., 1.]), np.array([0., 1.])]

# joint = get_joint_strategy_from_marginals(ne)
# joint = general_get_joint_strategy_from_marginals(ne)
# print(joint)

probs = [np.array([[[1.]],

       [[0.]]]), np.array([[[0.],
        [1.]]]), np.array([[[0., 1.]]])]

result = probs[0] * probs[1] * probs[2]

print(result[0][1][1])

new_probs = general_get_joint_strategy_from_marginals(ne)
print(new_probs[0][1][1])
