'''
Configuración de los directorios del módulo.
'''

import os
from platformdirs import user_data_dir, user_cache_dir, user_log_dir

appname = "evalia"
author = "ULPGC"

_data_dir = None
_cache_dir = None 
_log_dir = None

def data_dir():
    global _data_dir
    if _data_dir:
        return _data_dir
    elif os.getenv("EVALIA_DATA_DIR"):
        _data_dir = os.getenv("EVALIA_DATA_DIR")
    else:
        _data_dir = user_data_dir(appname, author)
    if not os.path.exists(_data_dir):
        os.makedirs(_data_dir)
    return _data_dir
    
def cache_dir():
    global _cache_dir
    if _cache_dir:
        return _cache_dir
    elif os.getenv("EVALIA_CACHE_DIR"):
        _cache_dir = os.getenv("EVALIA_CACHE_DIR")
    else:
        _cache_dir = user_cache_dir(appname, author)
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)
    return _cache_dir
    
def log_dir():
    global _log_dir
    if _log_dir:
        return _log_dir
    elif os.getenv("EVALIA_LOG_DIR"):
        _log_dir = os.getenv("EVALIA_LOG_DIR")
    else:
        _log_dir = user_log_dir(appname, author)
    if not os.path.exists(_log_dir):
        os.makedirs(_log_dir)
    return _log_dir
