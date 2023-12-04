import numpy as np

def multiplicative_weights_update(meta_games, num_iterations, step_size=None, alternate=False):
    """
    Implement the multiplicative weights update for two-player games only.
    :param meta_games: a list of payoffs, one for each player.
    :param num_iterations: Number of iterations for MWU.
    :param step_size: The value of step size for MWU.
    :param alternate: Make the updates alternate between players.
    :return: A mixed strategy profile in a list.
    """
    #TODO: output averaged strategies rather than pointwise.
    num_players = len(meta_games)
    n, m = np.shape(meta_games[0])

    if step_size is None:
        step_size = min(np.sqrt(8 * np.log(n)/num_iterations), np.sqrt(8 * np.log(m)/num_iterations))

    last_weights = [np.ones(n)/n, np.ones(m)/m]
    weights = [np.ones(n) / n, np.ones(m) / m]
    for i in range(num_iterations):
        if alternate:
            used_weights = weights
        else:
            used_weights = last_weights

        new_weights = weights[0] * np.exp(step_size * np.squeeze(np.dot(meta_games[0], used_weights[1])))
        weights[0] = new_weights / np.sum(new_weights)

        new_weights = weights[1] * np.exp(step_size * np.squeeze(np.dot(np.reshape(used_weights[0], (1,-1)), meta_games[1])))
        weights[1] = new_weights / np.sum(new_weights)

        if not alternate:
            last_weights[0] = np.copy(weights[0])
            last_weights[1] = np.copy(weights[1])

    for player in range(num_players):
        weights[player] /= np.sum(weights[player])

    return weights



# meta_games = [np.array([[8,6], [-8,7]]), np.array([[6,0], [-8,7]])]
# meta_games = [np.array([[ 4.,  2.],
#        [ 6., -6.]]), np.array([[ 2., -6.],
#        [ 6., -8.]])]
#
# weights = multiplicative_weights_update(meta_games, 50, alternate=False)
#
# print(weights)



