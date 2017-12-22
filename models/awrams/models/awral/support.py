import json
import os


def get_input_keys():
    '''
    Return the list of keys required as inputs
    '''
    from .settings import INPUT_JSON_FILE

    in_map = json.load(open(INPUT_JSON_FILE))

    model_keys = []

    model_keys += list(in_map['INPUTS_CELL'])
    model_keys += ['init_' + k for k in in_map['STATES_CELL']]
    model_keys += list(in_map['INPUTS_HYPSO'])

    for hru in ('_hrusr','_hrudr'):
        model_keys += ['init_' +k+hru for k in in_map['STATES_HRU']]
        model_keys += [k+hru for k in in_map['INPUTS_HRU']]

    return model_keys

def get_state_keys():
    in_map = get_input_meta()

    model_keys = []
    model_keys += [k for k in in_map['STATES_CELL']]
    for hru in ('_hrusr','_hrudr'):
        model_keys += [k+hru for k in in_map['STATES_HRU']]

    return model_keys

def get_input_meta():
    from .settings import INPUT_JSON_FILE

    in_map = json.load(open(INPUT_JSON_FILE))
    return in_map