# from data.pathways.pathway_loader import get_pathway_files
import itertools
import logging

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from keras.regularizers import l2

# from data.pathways.pathway_loader import get_pathway_files
from model.layers_custom import Diagonal, SparseTF

def shuffle_genes_map(mapp):
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # logging.info('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp


def get_pnet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not type(w_reg) == list:
        w_reg = [w_reg] * 10

    if not type(w_reg_outcomes) == list:
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not type(dropout) == list:
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}
    if non_neg:
        from keras.constraints import nonneg
        constraints = {'kernel_constraint': nonneg()}
        # constraints= {'kernel_constraint': nonneg(), 'bias_constraint':nonneg() }
    if sparse:
        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])
            logging.info('ones_ratio random {}'.format(ones_ratio))
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
            layer1 = SparseTF(n_genes, mapp, activation=activation, W_regularizer=reg_l(w_reg0),
                              name='h{}'.format(0), kernel_initializer=kernel_initializer, use_bias=use_bias,
                              **constraints)
            # layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg),
            #           use_bias=use_bias, name='h0', kernel_initializer= kernel_initializer )
        else:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)


    else:
        if sparse_first_layer:
            #
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
        else:
            layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                           use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)
    outcome = layer1(inputs)
    if attention:
        attention_probs = Diagonal(n_genes, input_shape=(n_features,), activation='sigmoid', W_regularizer=l2(w_reg0),
                                   name='attention0')(inputs)
        outcome = multiply([outcome, attention_probs], name='attention_mul')

    decision_outcomes = []

    # if reg_outcomes:
    # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), W_regularizer=reg_l(w_reg_outcome0), **constraints)(inputs)
    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), kernel_regularizer=reg_l(w_reg_outcome0))(
        inputs)
    # else:
    #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0))(inputs)

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    # decision_outcome = Activation( activation=activation_decision, name='o{}'.format(0))(decision_outcome)

    # first outcome layer
    # decision_outcomes.append(decision_outcome)

    # if reg_outcomes:
    # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1), W_regularizer=reg_l(w_reg_outcome1/2.), **constraints)(outcome)
    # else:
    #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1))(outcome)

    # drop2 = Dropout(dropout, name='dropout_{}'.format(0))
    drop2 = Dropout(dropout[0], name='dropout_{}'.format(0))

    outcome = drop2(outcome, training=dropout_testing)

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Activation(activation=activation_decision, name='o{}'.format(1))(decision_outcome)
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        layer_inds = list(range(1, len(maps)))
        # if adaptive_reg:
        #     w_regs = [float(w_reg)/float(i) for i in layer_inds]
        # else:
        #     w_regs = [w_reg] * len(maps)
        # if adaptive_dropout:
        #     dropouts = [float(dropout)/float(i) for i in layer_inds]
        # else:
        #     dropouts = [dropout]*len(maps)
        print('original dropout', dropout)
        print('dropout', layer_inds, dropout, w_reg)
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        print("mapp is:")
        print("maps[0]",len(maps[0]))
        print("maps[1]",len(maps[1]))
        print("maps[2]",len(maps[2]))
        print("maps[3]",len(maps[3]))
        print("maps[4]",len(maps[4]))
        print("maps[5]",len(maps[5]))
        print(maps[5])
        
#         print(pams.shape)
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            # dropout2 = dropouts[i]
            dropout = dropouts[1]
            names = mapp.index
            # names = list(mapp.index)
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
            # print 'map # ones {}'.format(np.sum(mapp))
            print('layer {}, dropout  {} w_reg {}'.format(i, dropout, w_reg))
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=reg_l(w_reg),
                                        name=layer_name, kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=reg_l(w_reg),
                                     name=layer_name, kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)

            if attention:
                attention_probs = Dense(n_pathways, activation='sigmoid', name='attention{}'.format(i + 1),
                                        W_regularizer=l2(w_reg))(outcome)
                outcome = multiply([outcome, attention_probs], name='attention_mul{}'.format(i + 1))

            # if reg_outcomes:
            # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2), W_regularizer=reg_l( w_reg2/(2**i)))(outcome)
            # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2), W_regularizer=reg_l( w_reg_outcome), **constraints)(outcome)
            decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2),
                                     kernel_regularizer=reg_l(w_reg_outcome))(outcome)
            # else:
            #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2))(outcome)
            # testing
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)
            drop2 = Dropout(dropout, name='dropout_{}'.format(i + 1))
            outcome = drop2(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names
            # feature_names.append(names)
        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
        # feature_names.append(maps[-1].index)
    return outcome, decision_outcomes, feature_names
