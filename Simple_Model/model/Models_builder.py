import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Lambda, Concatenate, Activation
from tensorflow.keras.optimizers.legacy import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from model.layers_custom import f1, Diagonal, SparseTF, TissueSpecific, GeneSelection
from model.model_utils import print_model, get_layers


# --------------------------------------------------------------------------------------------------
# Single Tissue: Input --> GeneSelection --> Output
def ST_I_G_O(data, maps, tissue_name, w_reg, learning_rate=.01, drop_rate=.3):
    print(f'\n\n\t\t{tissue_name}')
    print(f'Input --> GeneSelection --> Output')
    print(f'w_regs: {w_regs}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')
    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)

    ### Input layer  
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    mask = data.columns.levels[0].isin(maps[0].index)
    ### GeneSelection layer 
    GeneSelection_layer = GeneSelection(mask, n_status, name='GeneSelection')
    GeneSelection_layer_output = GeneSelection_layer(inputs)
    ### Dropout layer
    Dropout_layer = Dropout(rate=drop_rate, name='Dropout')
    Dropout_layer_output = Dropout_layer(GeneSelection_layer_output)
    
    ### 1 node Dense layer
    Dense_layer = Dense(1, kernel_regularizer=l2(w_reg), activation='sigmoid', name='Dense')
    outcome = Dense_layer(Dropout_layer_output)
    
    model = Model(inputs=inputs, outputs=outcome, name=f'ST_I_G_O_Model_{tissue_name}')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1])
    print(model.summary())
    return model


# --------------------------------------------------------------------------------------------------
# Single Tissue: Input --> GeneSelection --> Diagonal --> Output
def ST_I_G_D_O(data, maps, tissue_name, w_regs, learning_rate=.01, drop_rate=0.3):
    print(f'\n\n\t\t{tissue_name}')
    print(f'Input --> GeneSelection --> Diagonal --> Output')
    print(f'w_regs: {w_regs}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')
    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)

    ### Input layer  
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    mask = data.columns.levels[0].isin(maps[0].index)
    ### GeneSelection layer 
    GeneSelection_layer = GeneSelection(mask, n_status, name='GeneSelection')
    GeneSelection_layer_output = GeneSelection_layer(inputs)
    ### Diagonal
    Diagonal_layer = Diagonal(n_genes, input_shape=(n_features,), activation='tanh',
                              name='Diagonal',W_regularizer=l2(w_regs[0]))
    Diagonal_layer_output = Diagonal_layer(GeneSelection_layer_output)
    ### Dropout layer
    Dropout_layer = Dropout(rate=drop_rate, name='Dropout')
    Dropout_layer_output = Dropout_layer(Diagonal_layer_output)

    ### 1 node Dense layer
    Dense_layer = Dense(1, kernel_regularizer=l2(w_regs[1]), activation='sigmoid', name='Dense')
    outcome = Dense_layer(Dropout_layer_output)

    model = Model(inputs=inputs, outputs=outcome, name=f'ST_I_G_D_O_Model_{tissue_name}')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=f1)
    print(model.summary())
    return model


# --------------------------------------------------------------------------------------------------
# Single Tissue: Input --> GeneSelection --> Hidden Dense(3) --> Output
def ST_I_G_HD3_O(data, maps, tissue_name, w_regs, hidden_layers=(100,50,10),
              learning_rate=.0001, drop_rate=[.5, .3, .2, .1]):
    print(f'\n\n\t\t{tissue_name}')
    print(f'Input --> GeneSelection --> Hidden Dense(3) --> Output')
    print(f'w_regs: {w_regs}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')

    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)
    reg_l = l2

    ### Input layer  
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    mask = data.columns.levels[0].isin(maps[0].index)
    ### GeneSelection layer 
    GeneSelection_layer = GeneSelection(mask, n_status, name='GeneSelection')
    GeneSelection_layer_output = GeneSelection_layer(inputs)
    ### Dropout layer
    Dropout_layer = Dropout(rate=drop_rate[0], name='Dropout_1')
    Dropout_layer_output = Dropout_layer(GeneSelection_layer_output)
    for idx, l in enumerate(hidden_layers):
        hidden_layer = Dense(l, kernel_regularizer=reg_l(w_regs[idx]),
                             activation='tanh', name=f'Hidden_{idx+1}')
        hidden_layer_output = hidden_layer(Dropout_layer_output)
        
        Dropout_layer = Dropout(rate=drop_rate[idx+1], name=f'Dropout_{idx+2}')
        Dropout_layer_output = Dropout_layer(hidden_layer_output)
        
    ### 1 node Dense layer
    Dense_layer = Dense(1, kernel_regularizer=reg_l(w_regs[-1]), activation='sigmoid', name='Outcome')
    outcome = Dense_layer(Dropout_layer_output)
    
    model = Model(inputs=inputs, outputs=outcome, name=f'ST_I_G_HD3_O_Model_{tissue_name}')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=f1)
    print(model.summary())
    return model


# --------------------------------------------------------------------------------------------------
# Single Tissue: Input --> GeneSelection --> Diagonal --> Hidden Sparse(5) --> Output
def ST_I_G_D_HS5_O(data, maps, tissue_name, w_regs=[.1,.01,.01,.01,.01,.01,.001],
                learning_rate=.0001, drop_rate=[.5, .4, .2, .2, .1, .1]):
    print(f'\n\n\t\t{tissue_name}')
    print(f'Input --> GeneSelection --> Diagonal --> Hidden Sparse(5) --> Output')
    print(f'w_regs: {w_regs}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')

    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)

    ### Input layer  
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    mask = data.columns.levels[0].isin(maps[0].index)
    ### GeneSelection layer 
    GeneSelection_layer = GeneSelection(mask, n_status, name='GeneSelection')
    GeneSelection_layer_output = GeneSelection_layer(inputs)
    ### Diagonal
    Diagonal_layer = Diagonal(n_genes, input_shape=(n_features,), activation='tanh',
                              name='Diagonal', kernel_initializer='lecun_uniform', W_regularizer=l2(w_regs[0]))
    Diagonal_layer_output = Diagonal_layer(GeneSelection_layer_output)
    ### Dropout layer
    Dropout_layer = Dropout(rate=drop_rate[0], name='Dropout_1')
    Dropout_layer_output = Dropout_layer(Diagonal_layer_output)
    
    ### Hidden Sparse Layers
    for idx, mapp in enumerate(maps[0:-1]):
        n_genes, n_pathways = mapp.shape
        hidden_layer = SparseTF(n_pathways, mapp, activation='tanh', name=f'Hidden_{idx+1}',
                                W_regularizer=l2(w_regs[idx+1]), kernel_initializer='lecun_uniform')
        hidden_layer_output = hidden_layer(Dropout_layer_output)

        Dropout_layer = Dropout(rate=drop_rate[idx+1], name=f'Dropout_{idx+2}')
        Dropout_layer_output = Dropout_layer(hidden_layer_output)
    
    ### 1 node Dense layer
    Dense_layer = Dense(1, kernel_regularizer=l2(w_regs[-1]), activation='sigmoid', name='Outcome')
    outcome = Dense_layer(Dropout_layer_output)
    model = Model(inputs=inputs, outputs=outcome, name=f'ST_I_G_D_HS5_O_Model_{tissue_name}')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=f1)
    print(model.summary())
    return model

            
# --------------------------------------------------------------------------------------------------
# Single Tissue: Input --> GeneSelection --> Diagonal --> Sparse(5) --> Output(7)
def ST_I_G_D_HS5_O7(data, maps, tissue_name, w_regs=[0]*7,
                 w_regs_outcome=[0]*7, learning_rate=.0001,
                 drop_rate=[.5, .4, .2, .2, .1, .1],
                 loss_weights=[1, 3, 9, 27, 81, 243, 729]):
    
    print(f'\n\n\t\t{tissue_name}')
    print(f'Input --> GeneSelection --> Diagonal --> Sparse(5) --> Output(7)')
    print(f'w_regs: {w_regs}')
    print(f'w_regs_outcome: {w_regs_outcome}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')
    print(f'loss_weights: {loss_weights}')

    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)
    outcomes = []

    ### Input layer
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    mask = data.columns.levels[0].isin(maps[0].index)
    ### GeneSelection layer
    GeneSelection_layer = GeneSelection(mask, n_status, name='GeneSelection')
    GeneSelection_layer_output = GeneSelection_layer(inputs)
    ### Outcome from GeneSelection layer
    Outcome = Dense(1, activation='sigmoid', name='Outcome_1',
                    kernel_regularizer=l2(w_regs_outcome[0]))(GeneSelection_layer_output)
    outcomes.append(Outcome)
    ### Diagonal
    Diagonal_layer = Diagonal(n_genes, input_shape=(n_features,), activation='tanh',
                              name='Diagonal', kernel_initializer='lecun_uniform',
                              W_regularizer=l2(w_regs[0]))
    Diagonal_layer_output = Diagonal_layer(GeneSelection_layer_output)
    ### Outcome from Diagonal layer Outcome
    Outcome = Dense(1, activation='sigmoid', name='Outcome_2',
                    kernel_regularizer=l2(w_regs_outcome[1]))(Diagonal_layer_output)
    outcomes.append(Outcome)
    ### Dropout layer
    Dropout_layer = Dropout(rate=drop_rate[0], name='Dropout_1')
    Dropout_layer_output = Dropout_layer(Diagonal_layer_output)
    
    for idx, mapp in enumerate(maps[0:-1]):
        n_genes, n_pathways = mapp.shape
        ### Hidden Sparse Layers
        hidden_layer = SparseTF(n_pathways, mapp, activation='tanh', name=f'Hidden_{idx+1}',
                                W_regularizer=l2(w_regs[idx+1]), kernel_initializer='lecun_uniform')
        hidden_layer_output = hidden_layer(Dropout_layer_output)
        ### Outcome from Sparse layer
        Outcome = Dense(1, activation='sigmoid', name=f'Outcome_{idx+3}',
                        kernel_regularizer=l2(w_regs_outcome[idx+2]))(hidden_layer_output)
        outcomes.append(Outcome)
        ### Dropout layer
        if idx+1 < len(maps[:-1]):
            Dropout_layer = Dropout(rate=drop_rate[idx+1], name=f'Dropout_{idx+2}')
            Dropout_layer_output = Dropout_layer(hidden_layer_output)
    ### Buildining model
    n_outputs = len(outcomes)
    model = Model(inputs=[inputs], outputs=outcomes, 
                  name=f'ST_I_G_D_HS5_O7_Model_{tissue_name}')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy']*n_outputs,
                  metrics=[f1],
                  loss_weights=loss_weights)
    print(model.summary())
    return model


# --------------------------------------------------------------------------------------------------
# Single Tissue: Input --> GeneSelection --> Diagonal --> Sparse(5) --> Outcome(7) --> Output
def ST_I_G_D_HS5_O7_O(data, maps, tissue_name, w_regs=[0]*7,
                   w_regs_outcome=[0]*7, learning_rate=.0001,
                   drop_rate=[.5, .4, .2, .2, .1, .1]):
    
    print(f'\n\n\t\t{tissue_name}')
    print(f'Input --> GeneSelection --> Diagonal --> Sparse(5) --> Output(7) --> Output')
    print(f'w_regs: {w_regs}')
    print(f'w_regs_outcome: {w_regs_outcome}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')
    
    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)
    outcomes = []

    ### Input layer
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    ### making a mask 
    mask = data.columns.levels[0].isin(maps[0].index)
    ### GeneSelection layer
    GeneSelection_layer = GeneSelection(mask, n_status, name='GeneSelection')
    GeneSelection_layer_output = GeneSelection_layer(inputs)
    ### Outcome from GeneSelection layer
    Outcome = Dense(1, activation='linear', name='Outcome_1',
                    kernel_regularizer=l2(w_regs_outcome[0]))(GeneSelection_layer_output)
    outcomes.append(Outcome)
    ### Diagonal
    Diagonal_layer = Diagonal(n_genes, input_shape=(n_features,), activation='tanh',
                              name='Diagonal', kernel_initializer='lecun_uniform',
                              W_regularizer=l2(w_regs[0]))
    Diagonal_layer_output = Diagonal_layer(GeneSelection_layer_output)
    ### Outcome from Diagonal layer Outcome
    Outcome = Dense(1, activation='linear', name='Outcome_2',
                    kernel_regularizer=l2(w_regs_outcome[1]))(Diagonal_layer_output)
    outcomes.append(Outcome)
    ### Dropout layer
    Dropout_layer = Dropout(rate=drop_rate[0], name='Dropout_1')
    Dropout_layer_output = Dropout_layer(Diagonal_layer_output)
    
    for idx, mapp in enumerate(maps[0:-1]):
        n_genes, n_pathways = mapp.shape
        ### Hidden Sparse Layers
        hidden_layer = SparseTF(n_pathways, mapp, activation='tanh', name=f'Hidden_{idx+1}',
                                W_regularizer=l2(w_regs[idx+1]), kernel_initializer='lecun_uniform')
        hidden_layer_output = hidden_layer(Dropout_layer_output)
        ### Outcome from Sparse layer
        Outcome = Dense(1, activation='linear', name=f'Outcome_{idx+3}',
                        kernel_regularizer=l2(w_regs_outcome[idx+2]))(hidden_layer_output)
        outcomes.append(Outcome)
        ### Dropout layer
        if idx+1 < len(maps[:-1]):
            Dropout_layer = Dropout(rate=drop_rate[idx+1], name=f'Dropout_{idx+2}')
            Dropout_layer_output = Dropout_layer(hidden_layer_output)
    
    concatenated_outputs = Concatenate(axis=-1, name='Concatenate_outcomes')(outcomes)
    final_output = Dense(1, activation='sigmoid',name='Final_outcome')(concatenated_outputs)
    ### Buildining model
    model = Model(inputs=[inputs], outputs=final_output, name=f'ST_I_G_D_HS5_O7_O_Model_{tissue_name}')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'],
                  metrics=[f1])
    print(model.summary())
    return model

# --------------------------------------------------------------------------------------------------
# Multiple Tissues: Input --> GeneSelection --> Diagonal --> Hidden Sparse(5) --> Output
def MT_I_G_D_HS5_O7_O(data, tissues_maps, w_regs=[0]*7,
                      w_regs_outcome=[0]*7, learning_rate=.0001,
                      drop_rate=[.5, .4, .2, .2, .1, .1]):
    tissues_name = list(tissues_maps.keys())
    print(f'{tissues_name}')
    print(f'Input --> GeneSelection --> Diagonal --> Sparse(5) --> Output(7) --> Output')
    print(f'w_regs: {w_regs}')
    print(f'w_regs_outcome: {w_regs_outcome}')
    print(f'drop_rate: {drop_rate}')
    print(f'learning_rate: {learning_rate}')
    
    features = [f'{g}_{s}' for g,s in data.columns]
    n_features = len(features)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    status = data.columns.levels[1]
    n_status = len(status)
    ### Input layer
    inputs = Input(shape=(n_features,), dtype='float32', name=f'input')
    
    tissues_outcome = []
    for t, maps in tissues_maps.items():
        print(t)
        outcomes = []
        ### making a mask 
        mask = data.columns.levels[0].isin(maps[0].index)
        print(len(mask))
        ### GeneSelection layer
        GeneSelection_layer = GeneSelection(mask, n_status, name=f'GeneSelection-{t}')
        GeneSelection_layer_output = GeneSelection_layer(inputs)
        print(GeneSelection_layer_output.shape)
        ### Outcome from GeneSelection layer
        Outcome = Dense(1, activation='linear', name=f'Outcome_1-{t}',
                        kernel_regularizer=l2(w_regs_outcome[0]))(GeneSelection_layer_output)
        outcomes.append(Outcome)
        ### Diagonal
        Diagonal_layer = Diagonal(n_genes, input_shape=(n_features,), activation='tanh',
                                  name=f'Diagonal-{t}', kernel_initializer='lecun_uniform',
                                  W_regularizer=l2(w_regs[0]))
        Diagonal_layer_output = Diagonal_layer(GeneSelection_layer_output)
        ### Outcome from Diagonal layer Outcome
        Outcome = Dense(1, activation='linear', name=f'Outcome_2-{t}',
                        kernel_regularizer=l2(w_regs_outcome[1]))(Diagonal_layer_output)
        outcomes.append(Outcome)
        ### Dropout layer
        Dropout_layer = Dropout(rate=drop_rate[0], name=f'Dropout_1-{t}')
        Dropout_layer_output = Dropout_layer(Diagonal_layer_output)
        for idx, mapp in enumerate(maps[0:-1]):
            _, n_pathways = mapp.shape
            ### Hidden Sparse Layers
            hidden_layer = SparseTF(n_pathways, mapp, activation='tanh', name=f'Hidden_{idx+1}-{t}',
                                  W_regularizer=l2(w_regs[idx+1]),kernel_initializer='lecun_uniform')
            hidden_layer_output = hidden_layer(Dropout_layer_output)
            ### Outcome from Sparse layer
            Outcome = Dense(1, activation='linear', name=f'Outcome_{idx+3}-{t}',
                            kernel_regularizer=l2(w_regs_outcome[idx+2]))(hidden_layer_output)
            outcomes.append(Outcome)
            ### Dropout layer
            if idx+1 < len(maps[:-1]):
                Dropout_layer = Dropout(rate=drop_rate[idx+1], name=f'Dropout_{idx+2}-{t}')
                Dropout_layer_output = Dropout_layer(hidden_layer_output)

        concatenated_outputs = Concatenate(axis=-1, name=f'Concatenate_outcomes-{t}')(outcomes)
        final_output = Dense(1, activation='linear', name=f'Final_outcome-{t}')(concatenated_outputs)
        tissues_outcome.append(final_output)
    concatenated_outputs = Concatenate(axis=-1, name=f'Concatenate_tissues_outcomes')(tissues_outcome)
    tissues_output = Dense(1, activation='sigmoid',name=f'Final_tissues_outcome')(concatenated_outputs)
    model = Model(inputs=[inputs], outputs=tissues_output)
    print(model.summary())
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=['binary_crossentropy'], metrics=[f1])
    return model