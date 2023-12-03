import numpy as np
import itertools
from collections import deque

# a = np.array([[1,2,3],[4,5,6], [7,8,9]])
# b = tuple([slice(2), slice(2)])
#
# total_number_policies = [3, 3]
# number_new_policies = [1, 1]
#
# for current_player in range(2):
#
#     range_iterators = [
#               range(total_number_policies[k]) for k in range(current_player)
#           ] + [range(number_new_policies[current_player])] + [
#               range(total_number_policies[k])
#               for k in range(current_player + 1, 2)
#           ]
#     print(range_iterators)
#     print("====")
#     for current_index in itertools.product(*range_iterators):
#         print(current_index)
#
#     print("***")


a = [[123]]
print(a * 2)
