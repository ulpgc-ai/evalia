'''
Fábrica simple de objetos GPTManager. 
Proporciona un manejador global de GPT.
'''

from . import GPTManager
from . import GPTMockManager, GPTSmartManager, GPTBatchManager
from typing import Dict

mock_manager = GPTMockManager()
batch_managers: Dict[str, GPTBatchManager] = {}
smart_managers: Dict[str, GPTSmartManager] = {}

def get_manager(model, batch_api=False) -> GPTManager:
    if model == 'mock':
        return mock_manager
    if batch_api:
        if model in batch_managers:
            return batch_managers[model]
        manager = GPTBatchManager(model)
        batch_managers[model] = manager
        return manager
    else:
        if model in smart_managers:
            return smart_managers[model]
        manager = GPTSmartManager(model)
        smart_managers[model] = manager
        return manager
    

if __name__ == '__main__':
    pass