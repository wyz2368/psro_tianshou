from matrix_game_analysis.meta_strategies import double_oracle, fictitious_play, prd_solver
from matrix_game_analysis.psro4matrix_trainer import PSRO4matrix_trainer

def psro4matrix(meta_games,
         meta_strategy_string,
         checkpoint_dir,
         num_iterations,
         init_strategies=None):

    if meta_strategy_string == "DO":
        DO_trainer = PSRO4matrix_trainer(meta_games=meta_games,
                                       meta_method=double_oracle,
                                       checkpoint_dir=checkpoint_dir,
                                       num_iterations=num_iterations,
                                       init_strategies=init_strategies)
        DO_trainer.loop()

    elif meta_strategy_string == "FP":
        FP_trainer = PSRO4matrix_trainer(meta_games=meta_games,
                                       meta_method=fictitious_play,
                                       checkpoint_dir=checkpoint_dir,
                                       num_iterations=num_iterations,
                                       init_strategies=init_strategies)
        FP_trainer.loop()

    elif meta_strategy_string == "PRD":
        PRD_trainer = PSRO4matrix_trainer(meta_games=meta_games,
                                       meta_method=prd_solver,
                                       checkpoint_dir=checkpoint_dir,
                                       num_iterations=num_iterations,
                                       init_strategies=init_strategies)
        PRD_trainer.loop()

    else:
        raise ValueError("The specified meta strategy solver does not exist! Options: DO, FP, PRD")




