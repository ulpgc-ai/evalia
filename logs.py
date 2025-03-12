import logging

LOG_LEVEL = logging.DEBUG
LOG_FILE = 'app.log'

def get_logger(name):
    '''Common logger for all modules'''
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(message)s')
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
