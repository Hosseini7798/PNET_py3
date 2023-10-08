import keras 
import numpy 
import shap

# 
def is_startswith(string: str, chars: list) -> bool:
    res = False 
    for c in chars:
        res = res or string.startswith(c)
    return res

# note = make a class of getting shap value form any model
def get_layer_output(model: keras.Model,
                     x_train: numpy.ndarray,
                     x_test: numpy.ndarray) -> dict:
    layer_name = [l.name for l in model.layers]
    layer_shap = [l for l in layer_name if is_startswith(l, ['Gen', 'Dia', 'Hid'])]
    new_inputs_train = {}
    new_inputs_test = {}
    for l in layer_shap:
        desired_layer = model.get_layer(l)
        new_model = keras.Model(inputs=model.input,
                                outputs=desired_layer.output)
        new_inputs_train[l] = new_model.predict(x_train, verbose=0)
        new_inputs_test[l] = new_model.predict(x_test, verbose=0)
    return new_inputs_train, new_inputs_test

def get_layer_shap(model: keras.Model,
                   x_train: numpy.ndarray,
                   x_test: numpy.ndarray) -> dict:
    train_shap = {}
    test_shap = {}
    layer_name = [l.name for l in model.layers]
    layer_shap = [l for l in layer_name if is_startswith(l, ['Gen', 'Dia', 'Hid'])]
    layer_out = [l for l in layer_name if l.startswith('Out')]
    new_inputs_train, new_inputs_test = get_layer_output(model, x_train, x_test)

    if 'Final_tissues_outcome' in layer_name:
        list_inputs_train = [] 
        list_inputs_test = []
        inputs = []
        final_outcome = []
        tissues_name = [l.split('-')[1] for l in layer_shap if l.startswith('Gen')]
        layers = [l.split('-')[0] for l in layer_shap[::len(tissues_name)]]
        for t in tissues_name:
            outcomes = []
            print(t)
            layer_shap_tissue = [l for l in layer_shap if l.endswith(f'{t}')]
            layer_out_tissue = [l for l in layer_out if l.endswith(f'{t}')]
            list_inputs_train.extend([i for l, i in new_inputs_train.items() if l in layer_shap_tissue])
            list_inputs_test.extend([i for l, i in new_inputs_test.items() if l in layer_shap_tissue])
            for l, o in zip(layer_shap_tissue, layer_out_tissue):
                    new_inp = model.get_layer(l).output
                    inputs.append(new_inp)
                    new_out = model.get_layer(o)(new_inp)
                    outcomes.append(new_out)
            concatenated_outputs = model.get_layer(f'Concatenate_outcomes-{t}')(outcomes)
            final_output = model.get_layer(f'Final_outcome-{t}')(concatenated_outputs)
            final_outcome.append(final_output)
        concatenated_outputs = model.get_layer('Concatenate_tissues_outcomes')(final_outcome)
        final_output = model.get_layer('Final_tissues_outcome')(concatenated_outputs)
        new_model = keras.Model(inputs=inputs, outputs=final_output)
        explainer = shap.DeepExplainer(new_model, list_inputs_train)
        train_shap_values = explainer.shap_values(list_inputs_train)
        test_shap_values = explainer.shap_values(list_inputs_test)
        
        keys =[]
        for t in tissues_name:
            for l in layers:
                keys.append(f'{l}-{t}')
                
        for idx, i in enumerate(keys):
            train_shap[i] = train_shap_values[0][idx]
            test_shap[i] = test_shap_values[0][idx]
        
    elif 'Final_outcome' in layer_name:
        list_inputs_train = list(new_inputs_train.values())
        list_inputs_test = list(new_inputs_test.values())
        inputs = []
        outcomes = []
        for l, o in zip(layer_shap, layer_out):
            new_inp = model.get_layer(l).output
            inputs.append(new_inp)
            new_out = model.get_layer(o)(new_inp)
            outcomes.append(new_out)
        concatenated_outputs = model.get_layer('Concatenate_outcomes')(outcomes)
        final_output = model.get_layer('Final_outcome')(concatenated_outputs)
        new_model = keras.Model(inputs=inputs,
                                outputs=final_output)
        explainer = shap.DeepExplainer(new_model, list_inputs_train)
        train_shap_values = explainer.shap_values(list_inputs_train)
        test_shap_values = explainer.shap_values(list_inputs_test)
        for ldx, l in enumerate(layer_shap):
            train_shap[l] = train_shap_values[0][ldx]
            test_shap[l] = test_shap_values[0][ldx]
    else:    
        for l, o in zip(layer_shap, layer_out):
            print(f'{l} layer shap value is computing.')
            new_inp = model.get_layer(l).output
            new_out = model.get_layer(o)(new_inp)
            new_model = keras.Model(inputs=new_inp,
                                    outputs=new_out)
            explainer = shap.DeepExplainer(new_model, new_inputs_train[l])
            train_shap_values = explainer.shap_values(new_inputs_train[l])
            test_shap_values = explainer.shap_values(new_inputs_test[l])
            train_shap[l] =  train_shap_values
            test_shap[l] = test_shap_values
    return train_shap, test_shap