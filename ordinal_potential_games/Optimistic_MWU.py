import numpy as np

def optimistic_multiplicative_weights_update(meta_games, num_iterations, step_size, alternate=False):
    """
    Implement the optimistic multiplicative weights update for two-player games only.
    :param meta_games: two lists of payoffs, one for each player.
    :param num_iterations: Number of iterations for MWU.
    :param step_size: The value of step size for MWU.
    :param alternate: Make the updates alternate between players.
    :return: A mixed strategy profile in a list.
    """
    num_players = len(meta_games)
    n, m = np.shape(meta_games[0])

    if step_size is None:
        step_size = min(np.sqrt(8 * np.log(n)/num_iterations), np.sqrt(8 * np.log(m)/num_iterations))

    last_weights = [np.ones(n) / n, np.ones(m) / m]
    sec_last_weights = last_weights
    weights = [np.ones(n) / n, np.ones(m) / m]
    for i in range(num_iterations):
        if alternate:
            last_weights = used_weights
            used_weights = weights
        else:
            used_weights = weights

        new_weights = weights[0] * \
            np.exp(step_size * ( np.squeeze(2*np.dot(meta_games[0], used_weights[1]) - \
                                            np.dot(meta_games[0], last_weights[1]))))
        weights[0] = new_weights / np.sum(new_weights)

        new_weights = weights[1] * \
            np.exp(step_size * np.squeeze(2*np.dot(np.reshape(used_weights[0], (1,-1)), meta_games[1]) -\
                                          np.dot(np.reshape(last_weights[0], (1,-1)), meta_games[1])))
        weights[1] = new_weights / np.sum(new_weights)

        if not alternate:
            sec_last_weights[0] = last_weights[0]
            sec_last_weights[1] = last_weights[1]
            last_weights[0] = np.copy(used_weights[0])
            last_weights[1] = np.copy(used_weights[1])

    for player in range(num_players):
        weights[player] /= np.sum(weights[player])

    return weights