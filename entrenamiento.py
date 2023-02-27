from kerastuner import HyperParameters
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from kerastuner.tuners import RandomSearch
from modelo import *

def get_best_hyperparameters(train, n_users, n_items):

    # Define the search space
    hp = HyperParameters()

    hp.Choice('n_factors', [400, 500, 600])
    hp.Choice('learning_rate', [0.001])
    hp.Choice('dropout_rate', [0.4, 0.6, 0.8])
    hp.Choice('l2_regularizer', [0.001, 0.01, 0.05])
    hp.Int('layer_size', min_value=32, max_value=256, step=32)
    hp.Int('num_layers', min_value=1, max_value=5, step=1)
    hp.Choice('momentum', values=[0.1, 0.5, 0.9])
    hp.Choice('epsilon', values=[1e-06, 1e-05, 1e-04])
    hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'LeakyReLU', 'ELU', 'SELU'])

    # Define the tuner and perform the search
    tuner = RandomSearch(
        lambda hp: create_ncf_model(hp, n_users, n_items),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        hyperparameters=hp,
        overwrite=True
    )

    tuner.search([train['User'], train['Item']], train['Score'], epochs=100, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]
    
    return best_hps
