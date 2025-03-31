import logging
import os
from . import config

LOG_LEVEL = logging.DEBUG
LOG_FILENAME = 'app.log'

def log_file_path():
    return os.path.join(config.log_dir(), LOG_FILENAME)

def get_logger(name):
    '''Common logger for all modules'''
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_file_path())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    print(log_file_path())
