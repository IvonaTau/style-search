#!flask/bin/python

import os
import logging.config
import pickle
import yaml
import ctypes

def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

setup_logging()

mylib = ctypes.cdll.LoadLibrary('./libdarknetlnx.so')
print('libdarknet.so loaded')


from app import app

app.run(debug=True, host= '0.0.0.0', port = 3000, use_reloader=False)
