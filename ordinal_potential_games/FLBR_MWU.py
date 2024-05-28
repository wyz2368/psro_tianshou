import numpy as np

def FLBR_multiplicative_weights_update(meta_games, num_iterations, step_size=None, alternate=False):
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
            weights = last_weights
        
        int_weights = intermediate_step(meta_games, weights, step_size)
        
        new_weights = weights[0] * np.exp(step_size * np.squeeze(np.dot(meta_games[0], int_weights[1])))
        weights[0] = new_weights / np.sum(new_weights)

        new_weights = weights[1] * np.exp(step_size * np.squeeze(np.dot(np.reshape(int_weights[0], (1,-1)), meta_games[1])))
        weights[1] = new_weights / np.sum(new_weights)

        if not alternate:
            last_weights[0] = np.copy(weights[0])
            last_weights[1] = np.copy(weights[1])

    for player in range(num_players):
        weights[player] /= np.sum(weights[player])
    
    return weights
    
def intermediate_step(meta_games, weights, eta):
    tmp1 = np.multiply(weights[0], np.exp(eta*np.dot(meta_games[0], weights[1])))
    s_tmp1 = np.inner(weights[0], np.exp(eta*np.dot(meta_games[0], weights[1])))

    tmp2 = np.multiply(weights[1], np.exp(eta*np.dot(np.transpose(meta_games[1]), weights[0])))
    s_tmp2 = np.inner(weights[1], np.exp(eta*np.dot(np.transpose(meta_games[1]), weights[0])))
    
    return [tmp1/s_tmp1, tmp2/s_tmp2]