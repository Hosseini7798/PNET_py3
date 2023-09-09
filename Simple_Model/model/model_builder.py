import os 

import numpy as np
import pandas as pd 

from keras import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Lambda, Concatenate, Activation
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from model.layers_custom import f1, Diagonal, SparseTF
from model.model_utils import print_model, get_layers

from sklearn import metrics



# ---------------------------------------------------------------------------------------------------
# building a model 
def build_pnet(data, maps, optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True,
               loss_weights=1.0, activation_decision = 'linear',
               dropout=0.5, use_bias=False, activation='tanh', loss='binary_crossentropy',
               n_hidden_layers=1, kernel_initializer='glorot_uniform',
               dropout_testing=False, non_neg=False):
    feature_names = {}
    n_features = len(data.columns)
    genes = data.columns.levels[0]
    n_genes = len(genes)
    n_status = int(n_features/n_genes)

    inputs = Input(shape=(n_features,), dtype='float32', name=f'Gene_{n_genes}_Status_{n_status}')

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}
    if non_neg:
        from keras.constraints import nonneg
        constraints = {'kernel_constraint': nonneg()}
    layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation,
                      W_regularizer=l2(w_reg0), use_bias=use_bias, name='hidden_layer_1',
                      kernel_initializer=kernel_initializer, **constraints)
    outcome = layer1(inputs)
    print("#"*50)

    decision_outcomes = []
    decision_outcome = Dense(1, activation='linear', name='outcome_1',
                             kernel_regularizer=reg_l(w_reg_outcome0))(inputs)
    
    drop2 = Dropout(dropout[0], name='dropout_1')
    outcome = drop2(outcome, training=dropout_testing)
    
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        layer_inds = list(range(1, len(maps)))
        
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        print('(Genes with at least 1 conncetion to next layer)/(total conncetion to next layer)')
        print(f'({sum(maps[0].sum(axis=1)>0)})/({int(maps[0].values.sum())})')
        print(f'({sum(maps[1].sum(axis=1)>0)})/({int(maps[1].values.sum())})')
        print(f'({sum(maps[2].sum(axis=1)>0)})/({int(maps[2].values.sum())})')
        print(f'({sum(maps[3].sum(axis=1)>0)})/({int(maps[3].values.sum())})')
        print(f'({sum(maps[4].sum(axis=1)>0)})/({int(maps[4].values.sum())})')
        print(f'({sum(maps[5].sum(axis=1)>0)})/({int(maps[5].values.sum())})')
        print("#"*50)
        print("Map is:")
        print("Layer 1 conncetion:", maps[0].shape)
        print("Layer 2 conncetion:", maps[1].shape)
        print("Layer 3 conncetion:", maps[2].shape)
        print("Layer 4 conncetion:", maps[3].shape)
        print("Layer 5 conncetion:", maps[4].shape)
        print("Layer 6 conncetion:", maps[5].shape)
        print("#"*50)

        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            dropout = dropouts[1]
            names = mapp.index
            mapp = mapp.values

            n_genes, n_pathways = mapp.shape
            layer_name = 'hidden_layer_{}'.format(i + 2)
            hidden_layer = SparseTF(n_pathways, mapp, activation=activation,
                                    W_regularizer=reg_l(w_reg),name=layer_name,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=use_bias, **constraints)

            outcome = hidden_layer(outcome)
            decision_outcome = Dense(1, activation='linear', name='outcome_{}'.format(i + 2),
                                     kernel_regularizer=reg_l(w_reg_outcome))(outcome)

            decision_outcomes.append(decision_outcome)
            drop2 = Dropout(dropout, name='dropout_{}'.format(i + 2))
            outcome = drop2(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names

        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
    
    
    concatenated_outputs = Concatenate(axis=-1,name='Concatenate_outcomes')(decision_outcomes)
    final_output = Dense(1, activation='sigmoid',name='final_outcome')(concatenated_outputs)
    model = Model(inputs=[inputs], outputs=final_output)
    print(model.summary())
    model.compile(optimizer=optimizer, loss=['binary_crossentropy'], metrics=[f1],
                  loss_weights=loss_weights)

    return model 


def get_callbacks(X_train, y_train, select_best_model=False, save_name='pnet', reduce_lr=False,
                  monitor='val_o6_f1',save_gradient=False,epoch=300,early_stop=False,lr=0.001,
                 reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50)):
    import datetime
    callbacks = []
    timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}-{0:%S}'.format(datetime.datetime.now())
    save_filename = os.path.join(save_name + str(os.getpid()) + timeStamp)

    if reduce_lr:
        reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                      patience=2, min_lr=0.000001, verbose=1, mode='auto')
        callbacks.append(reduce_lr)

    if select_best_model:
        saving_callback = ModelCheckpoint(save_filename, monitor=monitor, verbose=1,
                                          save_best_only=True, mode='max')
        callbacks.append(saving_callback)

    if save_gradient:
        saving_gradient = GradientCheckpoint(save_filename, feature_importance, X_train,
                                             y_train, epoch, feature_names, 
                                             period=period)
        callbacks.append(saving_gradient)

    if early_stop:
        early_stop = FixedEarlyStopping(monitors=[monitor], min_deltas=[0.0], patience=10,
                                        verbose=1, modes=['max'], baselines=[0.0])
        callbacks.append(early_stop)

    if reduce_lr_after_nepochs:
        def step_decay(epoch, init_lr, drop, epochs_drop):
            import math
            initial_lrate = init_lr
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        from functools import partial
        reduce_lr_drop = reduce_lr_after_nepochs['drop']
        reduce_lr_epochs_drop = reduce_lr_after_nepochs['epochs_drop']
        step_decay_part = partial(step_decay, init_lr=lr, drop=reduce_lr_drop,
                                  epochs_drop=reduce_lr_epochs_drop)
        lr_callback = LearningRateScheduler(step_decay_part, verbose=1)
        callbacks.append(lr_callback)
    return callbacks

# ---------------------------------------------------------------------------------------------------
def get_th(y_validate, pred_scores):
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    for th in thresholds:
        y_pred = pred_scores > th
        f1 = metrics.f1_score(y_validate, y_pred)
        precision = metrics.precision_score(y_validate, y_pred)
        recall = metrics.recall_score(y_validate, y_pred)
        accuracy = metrics.accuracy_score(y_validate, y_pred)
        score = {}
        score['accuracy'] = accuracy
        score['precision'] = precision
        score['f1'] = f1
        score['recall'] = recall
        score['th'] = th
        scores.append(score)
    ret = pd.DataFrame(scores)
    best = ret[ret.f1 == max(ret.f1)]
    th = best.th.values[0]
    print(f'Best threshold to maximize the F1-score: {th}\n')
    print(f'Metrics for best threshold:\n{best}')
    return ret