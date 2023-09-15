wregs = [0.001] * 7
wreg_outcomes = [0.01] * 6
base_dropout = 0.5
n_hidden_layers = 5
loss_weights = [2, 7, 20, 54, 148, 400]

model_params={'use_bias': True,
              'w_reg': wregs,
              'w_reg_outcomes': wreg_outcomes,
              'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
              'loss_weights': loss_weights,
              'optimizer': 'Adam',
              'activation': 'tanh',
              'kernel_initializer': 'lecun_uniform',
              'n_hidden_layers': n_hidden_layers,
              'dropout_testing': False}


fitting_params={'samples_per_epoch':10,
                'select_best_model':False,
                'monitor':'val_o6_f1',
                'verbose':2,
                'epoch':300,
                'shuffle':True,
                'batch_size':100,
                'save_name':'pnet',
                'debug':False,
                'save_gradient':False,
                'class_weight':None,
                'n_outputs':n_hidden_layers + 1,
                'prediction_output':'average',
                'early_stop':False,
                'reduce_lr':False,
                'reduce_lr_after_nepochs':dict(drop=0.25, epochs_drop=50),
                'lr':0.001,
                'max_f1':True}