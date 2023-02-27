from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2

def create_ncf_model(hp, n_users, n_items):
    # Define inputs
    user_input = keras.Input(shape=(1,))
    item_input = keras.Input(shape=(1,))
    
    # Define embeddings
    user_embed_mlp = layers.Embedding(n_users, hp['n_factors'], embeddings_regularizer=l2(hp['l2_regularizer']))(user_input)
    item_embed_mlp = layers.Embedding(n_items, hp['n_factors'], embeddings_regularizer=l2(hp['l2_regularizer']))(item_input)
    
    user_embed_gmf = layers.Embedding(n_users, hp['n_factors'], embeddings_regularizer=l2(hp['l2_regularizer']))(user_input)
    item_embed_gmf = layers.Embedding(n_items, hp['n_factors'], embeddings_regularizer=l2(hp['l2_regularizer']))(item_input)
    
    # Flatten embeddings
    user_flat_mlp = layers.Flatten()(user_embed_mlp)
    item_flat_mlp = layers.Flatten()(item_embed_mlp)
    
    user_flat_gmf = layers.Flatten()(user_embed_gmf)
    item_flat_gmf = layers.Flatten()(item_embed_gmf)
    
    # MLP branch
    mlp_concat = layers.Concatenate()([user_flat_mlp, item_flat_mlp])
    
    for i in range(hp['num_layers']):
        mlp_concat = layers.Dense(hp['layer_size'], activation=hp['activation'])(mlp_concat)
        mlp_concat = layers.Dropout(hp['dropout_rate'])(mlp_concat)
        
    # GMF branch
    gmf_multiply = layers.Multiply()([user_flat_gmf, item_flat_gmf])
    
    # Concatenate GMF and MLP branches
    concat = layers.Concatenate()([mlp_concat, gmf_multiply])
    
    # Define output layer
    output = layers.Dense(1, activation='linear')(concat)
    
    # Define model
    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    
    # Compile model
    model.compile(
        loss='mape',
        optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate'])
    )
    
    return model
