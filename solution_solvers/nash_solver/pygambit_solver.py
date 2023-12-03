import random
import pygambit as gbt
import numpy as np

"""
This function enables pygambit to apply to PSRO meta_games in Openspiel.
"""

def pygbt_solve_matrix_games(meta_games, method="lcp", mode="all"):
    game = gbt.Game.from_arrays(*meta_games)
    if method == "lcp":
        if len(meta_games) != 2:
            raise ValueError("The number of players should be 2 for lcp.")
        ne = gbt.nash.lcp_solve(game, rational=False)
    elif method == "gnm":
        ne = gbt.nash.gnm_solve(game)
    elif method == "lp":
        if len(meta_games) != 2:
            raise ValueError("The number of players should be 2 for lp.")
        if not np.any(meta_games[0] + meta_games[1]):
            raise ValueError("The game is not zero-sum.")
        ne = gbt.nash.lp_solve(game, rational=False)
    elif method == "enumpure":
        ne = gbt.nash.enumpure_solve(game)
    elif method == "enummixed":
        if len(meta_games) != 2:
            raise ValueError("The number of players should be 2 for enummixed.")
        ne = gbt.nash.enummixed_solve(game, rational=False)
    elif method == "logit":
        ne = gbt.nash.logit_solve(game)
    elif method == "liap":
        ne = gbt.nash.liap_solve(game)
    elif method == "ipa":
        ne = gbt.nash.ipa_solve(game)
    elif method == "simpdiv":
        ne = gbt.nash.simpdiv_solve(game)
    else:
        raise NotImplementedError("The gambit method has not been implemented.")

    for eq in ne:
        eq.normalize()

    # Transfer the MixedStrategyProfile to list of arrays.
    results = []
    for eq in ne:
        one_result = []
        for player in eq.game.players:
            strategy = eq[player]
            probs = np.array([strategy.profile[s] for s in strategy.player.strategies])
            one_result.append(probs)
        results.append(one_result)

    if mode == "all":
        return results
    elif mode == "one":
        return [random.choice(results)]
    else:
        raise ValueError("mode option should be either 'one' or 'all'.")


# meta_games = [np.array([[1,2,3], [4,5,6], [7,8,9]]), np.array([[1,2,3], [4,5,6], [7,8,9]])]
# meta_games = [np.array([[ 4.,  2.],
#        [ 6., -6.]]), np.array([[ 2., -6.],
#        [ 6., -8.]])]

# meta_games = [np.array([[2, 0], [0, 2]]), np.array([[2, 0], [0, 2]])]

# meta_games = [np.array([[0.5]]), np.array([[-0.5]])]
# ne = pygbt_solve_matrix_games(meta_games, method="gnm", mode="one")
#
# print(ne)

# meta_games = [np.ones((4,4,4,4)) + 1,
#               np.ones((4,4,4,4)),
#               np.ones((4,4,4,4)),
#               np.ones((4,4,4,4))]
#
#
# ne = pygbt_solve_matrix_games(meta_games, method="enumpure", mode="one")
