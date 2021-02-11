import re
import numpy as np
from pathlib import Path

# Solves './gym_cellular_automata/envs/forest_fire/'
forest_fire_dir = Path(__file__).parents[1]

FOREST_FIRE_CONFIG_FILE = forest_fire_dir / 'forest_fire_config.yaml'

def get_config_dict(file):
    import yaml
    yaml_file = open(file, 'r')
    yaml_content = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_content

def get_forest_fire_config_dict():
    config = get_config_dict(FOREST_FIRE_CONFIG_FILE)
    config['cell_type'] = parse_type(config['cell_type'])
    config['action_type'] = parse_type(config['action_type'])
    return config

def parse_type(type_str):
    def remove_whitespace(my_str):
        return re.sub(r'\s+', '', my_str)

    type_str = remove_whitespace(type_str)
    
    re_uint8  = re.compile(r'uint8' , re.IGNORECASE)
    re_uint16 = re.compile(r'uint16', re.IGNORECASE)
    re_uint32 = re.compile(r'uint32', re.IGNORECASE)
    re_uint64 = re.compile(r'uint64', re.IGNORECASE)
    re_uint   = re.compile(r'uint'  , re.IGNORECASE)
    
    re_int8  = re.compile(r'int8' , re.IGNORECASE)
    re_int16 = re.compile(r'int16', re.IGNORECASE)
    re_int32 = re.compile(r'int32', re.IGNORECASE)
    re_int64 = re.compile(r'int64', re.IGNORECASE)
    re_int   = re.compile(r'int'  , re.IGNORECASE)

    # Unsigned int
    if re.search(re_uint8, type_str):
        return   np.uint8
    
    elif re.search(re_uint16, type_str):
        return     np.uint16
    
    elif re.search(re_uint32, type_str):
        return     np.uint32
    
    elif re.search(re_uint64, type_str):
        return     np.uint64
    
    elif re.search(re_uint, type_str):
        return     np.uintc

    # int
    elif re.search(re_int8, type_str):
        return     np.int8
    
    elif re.search(re_int16, type_str):
        return     np.int16

    elif re.search(re_int32, type_str):
        return     np.int32

    elif re.search(re_int64, type_str):
        return     np.int64
    
    elif re.search(re_int, type_str):
        return     np.intc

    else:
        return int
