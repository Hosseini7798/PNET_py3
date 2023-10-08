import pickle
import logging
import os
import time
import pandas as pd

from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import metrics


#------------------------------------------------------------------------------------------------------ 
def save_model(model, filename):
    print('saving model in', filename)
    f = file(filename + '.pkl', 'wb')
    import sys
    sys.setrecursionlimit(100000)
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


#------------------------------------------------------------------------------------------------------ 
def load_model(file_name):
    f = file(file_name + '.pkl', 'rb')
    # theano.config.reoptimize_unpickled_function = False
    start = time.time()
    model = pickle.load(f)
    end = time.time()
    elapsed_time = end - start
    return model

#------------------------------------------------------------------------------------------------------ 
def print_model(model, level=1):
    for i, l in enumerate(model.layers):
        indent = '  ' * level + '-'
        if type(l) == Sequential:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))
            print_model(l, level + 1)
        else:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))

#------------------------------------------------------------------------------------------------------ 
def get_layers(model, level=1):
    layers = []
    for i, l in enumerate(model.layers):

        # indent = '  ' * level + '-'
        if type(l) == Sequential:
            layers.extend(get_layers(l, level + 1))
        else:
            layers.append(l)

    return layers


#------------------------------------------------------------------------------------------------------ 
from model.coef_weights_utils import get_gradient_weights, get_permutation_weights, get_weights_linear_model, \
    get_gradient_weights_with_repeated_output, get_weights_gradient_outcome, \
    get_deep_explain_scores, get_shap_scores, get_skf_weights
import numpy as np

def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):
    if feature_importance.startswith('skf'):
        coef_ = get_skf_weights(model, X_train, y_train, feature_importance)
        # pass
    elif feature_importance == 'loss_gradient':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=False, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'loss_gradient_signed':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=True, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'gradient_outcome':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=False, signed=False)
    elif feature_importance == 'gradient_outcome_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target=target, detailed=detailed,
                                             multiply_by_input=False, signed=True)
    elif feature_importance == 'gradient_outcome*input':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=False)
    elif feature_importance == 'gradient_outcome*input_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=True)

    elif feature_importance.startswith('deepexplain'):
        method = feature_importance.split('_')[1]
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)

    elif feature_importance.startswith('shap'):
        method = feature_importance.split('_')[1]
        coef_ = get_shap_scores(model, X_train, y_train, target, method_name=method, detailed=detailed)


    elif feature_importance == 'gradient_with_repeated_outputs':
        coef_ = get_gradient_weights_with_repeated_output(model, X_train, y_train, target)
    elif feature_importance == 'permutation':
        coef_ = get_permutation_weights(model, X_train, y_train)
    elif feature_importance == 'linear':
        coef_ = get_weights_linear_model(model, X_train, y_train)
    elif feature_importance == 'one_to_one':
        weights = model.layers[1].get_weights()
        switch_layer_weights = weights[0]
        coef_ = np.abs(switch_layer_weights)
    else:
        coef_ = None
    return coef_

#------------------------------------------------------------------------------------------------------ 
def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output

#------------------------------------------------------------------------------------------------------ 
def plot_history(history):
    his = history.history
    if his.get('lr'):
        lr = his.pop('lr')
    val_his = [i for i in his.keys() if i.__contains__('val')]
    train_his = [i for i in his.keys() if not i.__contains__('val')]
    n_his = len(train_his)
    if len(val_his) == len(train_his):
        plt.figure(figsize=(12, n_his*5))
        for idx, (t,v) in enumerate(zip(train_his, val_his)):    
            plt.subplot(n_his, 2, idx+1)
            plt.plot(his[t], label=t)
            plt.plot(his[v], label=v)
            plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for i in train_his:
            if i.__contains__('_loss'):
                plt.plot(his[i],label=i)
                plt.legend()
        plt.subplot(1, 2, 2)
        for i in train_his:
            if i.__contains__('_f1'):
                plt.plot(his[i],label=i)
                plt.legend()
        plt.tight_layout()
        plt.show()
    pass

#------------------------------------------------------------------------------------------------------ 
def get_th(y_validate, pred_scores, return_result=False):
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
    if return_result:
        return ret